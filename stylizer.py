import argparse
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy


def extract_sketch_cv2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


# --- Style Transfer ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imsave(tensor, filename):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    image.save(filename)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, x):
        G = self.gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)


def get_model_and_losses(cnn, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    normalization = transforms.Normalize(mean=normalization_mean, std=normalization_std)
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    content_losses = []
    style_losses = []

    model = nn.Sequential()
    model = model.to(device)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_" + name, content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module("style_loss_" + name, style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:i + 1]

    return model, style_losses, content_losses


def run_style_transfer(cnn, content_img, style_img, num_steps=300, style_weight=1e6, content_weight=1):
    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img)
    input_img = content_img.clone()
    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * style_weight + content_score * content_weight
            loss.backward()
            run[0] += 1
            return loss

        optimizer.step(closure)
    input_img.data.clamp_(0, 1)
    return input_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to AI-generated image')
    parser.add_argument('--style', required=True, help='Path to style image (e.g. monet.png)')
    parser.add_argument('--middle', default='middle.png', help='Path to save sketch')
    parser.add_argument('--output', default='result.jpg', help='Path to save styled result')
    args = parser.parse_args()

    # Шаг 1 — создаем эскиз
    img = cv2.imread(args.input)
    if img is None:
        raise ValueError(f"Cannot read {args.input}")
    sketch = extract_sketch_cv2(img)
    cv2.imwrite(args.middle, sketch)
    print(f"[✓] Saved sketch as {args.middle}")

    # Шаг 2 — загрузка изображений для переноса
    content_img = image_loader(args.middle)
    style_img = image_loader(args.style)

    # Шаг 3 — перенос
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    output = run_style_transfer(cnn, content_img, style_img)

    imsave(output, args.output)
    print(f"[✓] Saved styled image as {args.output}")


if __name__ == '__main__':
    main()
