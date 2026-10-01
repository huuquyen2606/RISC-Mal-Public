"""Generative replay and generative classification models for continual learning.

Implements:
1. LatentCVAE (LGR - Latent Generative Replay):
   - Conditional VAE modeling latent feature space R(x) conditioned on one-hot class vectors.
2. ConditionalVAE (BI-R - Brain-Inspired Replay):
   - Expandable Conditional VAE with per-class Gaussian Mixture Model (GMM) prior.
3. ClassVAE & GCHead (GC - Generative Classifier):
   - Modular class-specific VAEs predicting class logits via ELBO likelihood scoring.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 64


# =============================================================================
# 1. LATENT CVAE FOR LGR (Stoychev et al.)
# =============================================================================

class LatentCVAE(nn.Module):
    """Conditional Variational Autoencoder for Latent Generative Replay (LGR).

    Encodes 64-dimensional latent features concatenated with one-hot class indicators
    into a continuous latent distribution z ~ N(mu, sigma^2), and decodes back to
    reconstruct R(x) for replay rehearsal.

    Args:
        feature_dim: Dimensionality of latent features (default: 64).
        latent_dim: Dimensionality of VAE bottleneck z (default: 32).
        hidden_dim: Hidden layer capacity (default: 128).
        num_classes: Total number of classes across the benchmark (default: 6).
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Encoder: [feature_dim + num_classes] -> hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder: [latent_dim + num_classes] -> hidden_dim -> feature_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, feature_dim),
        )

    def _safe_bn_forward(self, seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """Protects BatchNorm1d against single-sample batch execution during training."""
        if self.training and x.size(0) == 1:
            bn = seq[2]
            was_training = bn.training
            bn.eval()
            out = seq(x)
            if was_training:
                bn.train()
            return out
        return seq(x)

    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes features and one-hot condition vector into mean and log-variance."""
        inp = torch.cat([x, c], dim=1)
        h = self._safe_bn_forward(self.encoder, inp)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decodes latent vector z and one-hot condition vector into feature space."""
        inp = torch.cat([z, c], dim=1)
        return self._safe_bn_forward(self.decoder, inp)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass returning (x_recon, mu, log_var)."""
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, c)
        return x_recon, mu, log_var

    def sample(
        self,
        num_samples: int,
        class_idx: int,
        device: Union[torch.device, str] = "cpu",
    ) -> torch.Tensor:
        """Generates synthetic replay features conditioned on a specific class label."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            c = torch.zeros(num_samples, self.num_classes, device=device)
            c[:, class_idx] = 1.0
            x_fake = self.decode(z, c)
        return x_fake

    @staticmethod
    def vae_loss(
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Evidence Lower Bound (ELBO) loss combining reconstruction MSE and KL divergence."""
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss


# =============================================================================
# 2. EXPANDABLE CONDITIONAL VAE FOR BI-R (Brain-Inspired Replay)
# =============================================================================

class ConditionalVAE(nn.Module):
    """Brain-Inspired Replay (BI-R) conditional VAE with expandable GMM prior.

    Features dynamically expanding condition vectors and learnable Gaussian Mixture
    Model (GMM) prior parameters (prior_mu, prior_log_var) per class.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        initial_classes: int = 2,
        hidden_dim: int = 256,
        z_dim: int = 64,
        gate_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = initial_classes
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        enc_in = input_dim + initial_classes
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_log_var = nn.Linear(hidden_dim, z_dim)

        dec_in = z_dim + initial_classes
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=gate_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=gate_dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        # Learnable GMM prior means and log-variances per class
        self.prior_mu = nn.Parameter(torch.zeros(initial_classes, z_dim))
        self.prior_log_var = nn.Parameter(torch.zeros(initial_classes, z_dim))

    def expand_classes(
        self,
        num_new_classes: int,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Expands condition vectors, encoder/decoder input matrices, and GMM priors."""
        if num_new_classes <= 0:
            return

        old_c = self.num_classes
        new_c = old_c + num_new_classes
        self.num_classes = new_c

        # 1. Expand learnable GMM prior parameters
        new_mu = nn.Parameter(torch.zeros(new_c, self.z_dim, device=device))
        new_lv = nn.Parameter(torch.zeros(new_c, self.z_dim, device=device))
        with torch.no_grad():
            new_mu[:old_c] = self.prior_mu
            new_lv[:old_c] = self.prior_log_var
        self.prior_mu = new_mu
        self.prior_log_var = new_lv

        # 2. Expand first linear layer of encoder
        old_enc = self.encoder[0]
        new_enc = nn.Linear(self.input_dim + new_c, old_enc.out_features).to(device)
        with torch.no_grad():
            new_enc.weight[:, : self.input_dim + old_c] = old_enc.weight
            new_enc.bias = old_enc.bias
        self.encoder[0] = new_enc

        # 3. Expand first linear layer of decoder
        old_dec = self.decoder[0]
        new_dec = nn.Linear(self.z_dim + new_c, old_dec.out_features).to(device)
        with torch.no_grad():
            new_dec.weight[:, : self.z_dim + old_c] = old_dec.weight
            new_dec.bias = old_dec.bias
        self.decoder[0] = new_dec

    def _one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        """Converts integer class labels to one-hot indicator vectors."""
        return F.one_hot(labels.long(), num_classes=self.num_classes).float()

    def encode(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes features and labels into latent distribution parameters."""
        c = self._one_hot(labels).to(x.device)
        inp = torch.cat([x, c], dim=1)
        h = self.encoder(inp)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Decodes latent z and label conditioning into reconstructed features."""
        c = self._one_hot(labels).to(z.device)
        inp = torch.cat([z, c], dim=1)
        return self.decoder(inp)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        mu, log_var = self.encode(x, labels)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, labels)
        return x_recon, mu, log_var

    def elbo_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes ELBO loss with class-specific GMM prior matching."""
        x_recon, mu, log_var = self.forward(x, labels)
        recon_loss = F.mse_loss(x_recon, x, reduction="none").sum(dim=1).mean()

        idx = labels.long()
        p_mu = self.prior_mu[idx]
        p_lv = self.prior_log_var[idx]

        kl = 0.5 * torch.sum(
            p_lv - log_var + (log_var.exp() + (mu - p_mu).pow(2)) / p_lv.exp() - 1.0,
            dim=1,
        ).mean()

        return recon_loss + kl

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        class_id: int,
        device: Union[torch.device, str] = "cpu",
    ) -> torch.Tensor:
        """Samples synthetic replay features from class-specific GMM prior."""
        self.eval()
        mu_c = self.prior_mu[class_id]
        lv_c = self.prior_log_var[class_id]
        std_c = torch.exp(0.5 * lv_c)
        z = mu_c + std_c * torch.randn(n_samples, self.z_dim, device=device)
        labels = torch.full((n_samples,), class_id, dtype=torch.long, device=device)
        return self.decode(z, labels)


# =============================================================================
# 3. CLASS-SPECIFIC VAE AND GENERATIVE CLASSIFIER HEAD FOR GC
# =============================================================================

class ClassVAE(nn.Module):
    """Independent single-class VAE modeling p(x | c) for a single malware family."""

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        z_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_log_var = nn.Linear(hidden_dim, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes class feature x into latent distribution parameters (mu, log_var)."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent sample z back to feature reconstruction."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (x_recon, mu, log_var)."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def elbo(self, x: torch.Tensor) -> torch.Tensor:
        """Computes negative ELBO (log-likelihood surrogate) for classification scoring."""
        x_recon, mu, log_var = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x, reduction="none").sum(dim=1)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return -(recon_loss + kl_loss)


class GCHead(nn.Module):
    """Generative Classifier (GC) head holding an ensemble of class-specific VAEs.

    During training: forwards through a linear backup layer.
    During evaluation: assigns logits equal to class-conditional ELBO log-likelihoods.
    """

    def __init__(
        self,
        initial_classes: int = 2,
        feature_dim: int = FEATURE_DIM,
        num_classes: Optional[int] = None,
        latent_dim: Optional[int] = None,
    ) -> None:
        if num_classes is not None:
            initial_classes = num_classes
        if latent_dim is not None:
            feature_dim = latent_dim
        super().__init__()
        self.num_classes = initial_classes
        self.feature_dim = feature_dim
        self.vaes: Dict[int, ClassVAE] = {}
        self.vae_modules = nn.ModuleList()
        self.linear_backup = nn.Linear(feature_dim, initial_classes)

    def expand_classes(
        self,
        num_new_classes: int,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Expands the linear backup layer for newly discovered classes."""
        old_out = self.linear_backup.out_features
        new_out = old_out + num_new_classes
        new_linear = nn.Linear(self.feature_dim, new_out).to(device)
        with torch.no_grad():
            new_linear.weight[:old_out, :] = self.linear_backup.weight
            new_linear.bias[:old_out] = self.linear_backup.bias
        self.linear_backup = new_linear
        self.num_classes = new_out

    def register_vae(self, class_id: int, vae: ClassVAE) -> None:
        """Registers a trained ClassVAE for a specific class."""
        self.vaes[class_id] = vae
        self.vae_modules.append(vae)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns linear logits if training, ELBO logits if in eval mode."""
        if self.training:
            return self.linear_backup(features)
        return self._elbo_logits(features)

    def _elbo_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Computes ELBO log-likelihood scores for each class."""
        b = features.shape[0]
        device = features.device
        scores = torch.full((b, self.num_classes), -1e9, device=device)
        for class_id, vae in self.vaes.items():
            if class_id < self.num_classes:
                vae.eval()
                scores[:, class_id] = vae.elbo(features)
        return scores
