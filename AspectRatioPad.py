import torchvision.transforms.functional as F

from PIL import Image, ImageFilter

class AspectRatioPad:
    """アスペクト比を維持してリサイズし、余白を黒で埋める"""
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, image):
        # 1. 元のサイズを取得
        w, h = image.size
        target_w, target_h = self.size

        # 2. 倍率を計算（アスペクト比維持）
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        # 3. リサイズ
        image = F.resize(image, (new_h, new_w))

        # 4. 黒塗りの土台（キャンバス）を作成
        new_image = Image.new("RGB", self.size, (0, 0, 0))

        # 5. 中心に貼り付け
        upper = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        new_image.paste(image, (left, upper))

        return new_image
