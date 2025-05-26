import numpy as np  # numpyを使用してπを取得
import torch


def compute_abc_angle(batch_lattice_vec: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Computes the side lengths and angles of crystal lattices from their lattice vectors for a batch.

    This function calculates the side lengths (a, b, c) and angles (alpha, beta, gamma) of the crystal lattices
    based on the provided lattice vectors for each item in the batch. The angles are computed in degrees.

    Args:
        batch_lattice_vec (torch.Tensor): A tensor containing the lattice vectors for the crystal lattices in the batch.
                                          The shape is (B, 3, 3), where B is the batch size.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                                           - The first tensor contains the side lengths (a, b, c) of the crystal lattices in the batch, with shape (B, 3).
                                           - The second tensor contains the angles (alpha, beta, gamma) in degrees of the crystal lattices in the batch, with shape (B, 3).
    """
    B = batch_lattice_vec.shape[0]
    batch_abc = torch.zeros(
        (B, 3), dtype=batch_lattice_vec.dtype, device=batch_lattice_vec.device
    )
    batch_angle = torch.zeros(
        (B, 3), dtype=batch_lattice_vec.dtype, device=batch_lattice_vec.device
    )

    # 辺の長さを計算
    batch_abc[:, 0] = torch.norm(batch_lattice_vec[:, 0, :], dim=1)
    batch_abc[:, 1] = torch.norm(batch_lattice_vec[:, 1, :], dim=1)
    batch_abc[:, 2] = torch.norm(batch_lattice_vec[:, 2, :], dim=1)

    # 成す角を計算
    # alpha = angle between b and c
    batch_angle[:, 0] = torch.acos(
        torch.sum(batch_lattice_vec[:, 1, :] * batch_lattice_vec[:, 2, :], dim=1)
        / (batch_abc[:, 1] * batch_abc[:, 2])
    )
    # beta = angle between a and c
    batch_angle[:, 1] = torch.acos(
        torch.sum(batch_lattice_vec[:, 0, :] * batch_lattice_vec[:, 2, :], dim=1)
        / (batch_abc[:, 0] * batch_abc[:, 2])
    )
    # gamma = angle between a and b
    batch_angle[:, 2] = torch.acos(
        torch.sum(batch_lattice_vec[:, 0, :] * batch_lattice_vec[:, 1, :], dim=1)
        / (batch_abc[:, 0] * batch_abc[:, 1])
    )

    # ラジアンから度へ変換
    batch_angle = torch.rad2deg(batch_angle)

    return batch_abc, batch_angle


def compute_lattice_vectors(
    batch_abc: torch.Tensor, batch_angle: torch.Tensor
) -> torch.Tensor:
    """
    Computes the lattice vectors for a batch of crystal lattices based on their side lengths and angles.

    This function calculates the lattice vectors for each set of crystal lattice parameters in the batch. The lattice
    vectors are computed using the side lengths (a, b, c) and angles (alpha, beta, gamma) provided for each crystal lattice.

    Args:
        batch_abc (torch.Tensor): A tensor containing the side lengths of the crystal lattices in the batch.
                                  The shape is (B, 3), where B is the batch size.
        batch_angle (torch.Tensor): A tensor containing the angles (in degrees) of the crystal lattices in the batch.
                                    The shape is (B, 3), where B is the batch size. Angles are in the order (alpha, beta, gamma).

    Returns:
        torch.Tensor: A tensor of shape (B, 3, 3) containing the lattice vectors for each crystal lattice in the batch.
    """
    # 角度をラジアンに変換
    angle_rad = batch_angle * np.pi / 180.0

    # 成す角のコサインとサイン
    cos_alpha = torch.cos(angle_rad[:, 0])
    cos_beta = torch.cos(angle_rad[:, 1])
    cos_gamma = torch.cos(angle_rad[:, 2])
    sin_gamma = torch.sin(angle_rad[:, 2])

    # バッチサイズの取得
    B = batch_abc.shape[0]

    # 格子ベクトルテンソルの初期化
    lattice_vec = torch.zeros((B, 3, 3), dtype=batch_abc.dtype, device=batch_abc.device)

    # 格子ベクトルの計算
    lattice_vec[:, 0, 0] = batch_abc[:, 0]  # a
    lattice_vec[:, 1, 0] = batch_abc[:, 1] * cos_gamma  # b*cos(gamma)
    lattice_vec[:, 1, 1] = batch_abc[:, 1] * sin_gamma  # b*sin(gamma)
    lattice_vec[:, 2, 0] = batch_abc[:, 2] * cos_beta  # c*cos(beta)
    lattice_vec[:, 2, 1] = (
        batch_abc[:, 2] * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    )
    lattice_vec[:, 2, 2] = batch_abc[:, 2] * torch.sqrt(
        1 - cos_beta**2 - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma) ** 2
    )

    return lattice_vec


def calculate_bond_cosine(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """calculate cosine
    Args:
        r1: shape=(N,3)
        r2: shape=(N,3)
    """
    bond_cosine = torch.nn.functional.cosine_similarity(r1, r2, dim=1)
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine


def direct_to_cartesian(
    dir_coords: torch.Tensor, lattice_vectors: torch.Tensor
) -> torch.Tensor:
    """
    Convert direct coordinates to cartesian coordinates using lattice vectors.

    The conversion is done by matrix multiplying the direct coordinates with the lattice vectors.

    Args:
        dir_coords (torch.Tensor): The direct coordinates, with shape (N, 3), where N is the number
                                   of atoms in the structure.
        lattice_vectors (torch.Tensor): The lattice vectors, with shape (3, 3).

    Returns:
        torch.Tensor: The cartesian coordinates, with the same shape as dir_coords.
    """
    return torch.matmul(dir_coords, lattice_vectors)


def direct_to_cartesian_batch(
    dir_coords: torch.Tensor, lattice_vectors: torch.Tensor
) -> torch.Tensor:
    """
    Convert direct coordinates to cartesian coordinates using lattice vectors.

    The conversion is done by matrix multiplying the direct coordinates with the lattice vectors.

    Args:
        dir_coords (torch.Tensor): The direct coordinates, with shape (N, 3), where N is the number
                                   of atoms in the structure.
        lattice_vectors (torch.Tensor): The lattice vectors, with shape (N, 3, 3).

    Returns:
        torch.Tensor: The cartesian coordinates, with the same shape as dir_coords.
    """
    _bool1 = len(dir_coords.shape) == 2
    _bool2 = len(lattice_vectors.shape) == 3
    _bool3 = dir_coords.shape[1] == 3
    if not (_bool1 and _bool2 and _bool3):
        print("dir_coords.shape:", dir_coords.shape)
        print("lattice_vectors.shape:", lattice_vectors.shape)
        raise Exception(
            "The shapes of dir_coords and lattice_vectors are not compatible."
        )

    cartesian_coords = torch.bmm(dir_coords.unsqueeze(1), lattice_vectors)
    return cartesian_coords.squeeze(1)
