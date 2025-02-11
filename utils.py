import torch
import torch.nn.functional as F

def gaussian_filter(kernel_size=5, sigma=1.0, channels=1):
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
    g = torch.exp(-0.5 * (x / sigma)**2)
    g = g / g.sum()
    kernel = torch.ger(g, g)  # Outer product to create a 2D Gaussian filter
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add channel dimensions
    kernel = kernel.repeat(channels, 1, 1, 1)  # Repeat kernel for each channel
    return kernel

def ssim_single_scale(img1, img2, k1=0.01, k2=0.03, filter_size=3, filter_sigma=1.0, L=1.0):
    channels = img1.shape[1]  # Get the number of channels
    kernel = gaussian_filter(kernel_size=filter_size, sigma=filter_sigma, channels=channels).to(img1.device)

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    # Mean values
    mu1 = F.conv2d(img1, kernel, padding="same", groups=channels)
    mu2 = F.conv2d(img2, kernel, padding="same", groups=channels)

    # Variances and covariances
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 ** 2, kernel, padding="same", groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, padding="same", groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding="same", groups=channels) - mu1_mu2

    # SSIM calculation
    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return torch.clamp(ssim, 0, 1)

def ms_ssim(img1, img2, power_factors=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], filter_size=3, filter_sigma=1.0, L=1.0):
    img1 = img1.float()
    img2 = img2.float()

    msssim = []
    weights = torch.tensor(power_factors, dtype=torch.float32, device=img1.device)

    for weight in weights[:-1]:  # Loop over all but the last scale
        ssim_map = ssim_single_scale(img1, img2, filter_size=filter_size, filter_sigma=filter_sigma, L=L)
        msssim.append(weight * ssim_map.mean())

        # Downsample the images
        img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
        img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)

    # Compute SSIM for the final scale
    final_ssim = ssim_single_scale(img1, img2, filter_size=filter_size, filter_sigma=filter_sigma, L=L)
    msssim.append(weights[-1] * final_ssim.mean())

    return sum(msssim)
