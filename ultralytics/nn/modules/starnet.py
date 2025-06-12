import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
# Import hàm tiện ích make_divisible từ ultralytics
from ultralytics.utils.ops import make_divisible

__all__ = ['starnet_s050', 'starnet_s100', 'starnet_s150', 'starnet_s1', 'starnet_s2', 'starnet_s3', 'starnet_s4']

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        mlp_dim = make_divisible(mlp_ratio * dim, 8) # Đảm bảo mlp_dim chia hết cho 8
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_dim, 1, with_bn=False)
        self.g = ConvBN(mlp_dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarNet(nn.Module):
    # Thêm width_multiple vào __init__ với giá trị mặc định là 1.0 (không scale)
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, width_multiple=1.0, **kwargs):
        super().__init__()
        
        # <<< THAY ĐỔI QUAN TRỌNG >>>
        # Scale base_dim và in_channel dựa trên width_multiple
        # làm tròn đến bội số gần nhất của 8 để tối ưu phần cứng
        scaled_base_dim = make_divisible(base_dim * width_multiple, 8)
        self.in_channel = make_divisible(32 * width_multiple, 8)
        
        self.num_classes = num_classes
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            # Sử dụng scaled_base_dim đã được tính toán
            embed_dim = make_divisible(scaled_base_dim * 2 ** i_layer, 8)
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        
        # Xác định số kênh đầu ra P3, P4, P5
        # forward mẫu để lấy shape, đảm bảo kích thước ảnh đủ lớn
        dummy_input = torch.randn(1, 3, 640, 640)
        # Lấy 3 feature map cuối cùng tương ứng P3, P4, P5
        output_features = self.forward(dummy_input)
        self.channel = [f.size(1) for f in output_features[-3:]]
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)): # Sửa lại tuple
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)): # Sửa lại tuple
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        features = []
        x = self.stem(x)
        features.append(x) # P2/4
        for stage in self.stages:
            x = stage(x)
            features.append(x) # P3, P4, P5
        # Trả về P2, P3, P4, P5
        return features # Bỏ qua P1, P2 (stem)


# Sửa lại các hàm khởi tạo để xử lý việc tải pretrained weights
def _create_starnet(variant_name, base_dim, depths, mlp_ratio, pretrained=False, **kwargs):
    """Hàm trợ giúp để tạo model và xử lý pretrained weights."""
    # Lấy width_multiple từ kwargs, nếu không có thì mặc định là 1.0
    width_multiple = kwargs.get('width_multiple', 1.0)
    
    # Tạo model và truyền tất cả kwargs vào
    model = StarNet(base_dim, depths, mlp_ratio, **kwargs)
    
    # Chỉ tải pretrained weights nếu không scale và người dùng yêu cầu
    if pretrained and width_multiple == 1.0:
        if variant_name in model_urls:
            url = model_urls[variant_name]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
            # YOLOv8 thường dùng state_dict["model"], nhưng checkpoint này dùng "state_dict"
            # nên cần kiểm tra và load cho đúng
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            print(f"INFO: Loaded pretrained weights for {variant_name}")
        else:
            print(f"WARNING: No pretrained weights available for {variant_name}")
    elif pretrained and width_multiple != 1.0:
        print(f"INFO: Pretrained weights not loaded for {variant_name} because the model is scaled (width_multiple={width_multiple}).")
        
    return model

# Cập nhật các hàm khởi tạo model
def starnet_s1(pretrained=False, **kwargs):
    return _create_starnet('starnet_s1', 24, [2, 2, 8, 3], 4, pretrained, **kwargs)

def starnet_s2(pretrained=False, **kwargs):
    return _create_starnet('starnet_s2', 32, [1, 2, 6, 2], 4, pretrained, **kwargs)

def starnet_s3(pretrained=False, **kwargs):
    return _create_starnet('starnet_s3', 32, [2, 2, 8, 4], 4, pretrained, **kwargs)

def starnet_s4(pretrained=False, **kwargs):
    return _create_starnet('starnet_s4', 32, [3, 3, 12, 5], 4, pretrained, **kwargs)

# very small networks (không có pretrained weights nên giữ nguyên)
def starnet_s050(pretrained=False, **kwargs):
    return StarNet(16, [1, 1, 3, 1], 3, **kwargs)

def starnet_s100(pretrained=False, **kwargs):
    return StarNet(20, [1, 2, 4, 1], 4, **kwargs)

def starnet_s150(pretrained=False, **kwargs):
    return StarNet(24, [1, 2, 4, 2], 3, **kwargs)
