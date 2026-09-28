"""Password binary code generation (from FIT codebase)."""

import numpy as np
import torch


def unsigned_long_to_binary_repr(unsigned_long, passwd_length):
    batch_size = unsigned_long.shape[0]
    target_size = passwd_length // 4
    binary = np.empty((batch_size, passwd_length), dtype=np.float32)
    for idx in range(batch_size):
        binary[idx, :] = np.array(
            [int(item) for item in bin(unsigned_long[idx])[2:].zfill(passwd_length)])
    dis_target = np.empty((batch_size, target_size), dtype=np.int_)
    for idx in range(batch_size):
        tmp = unsigned_long[idx]
        for byte_idx in range(target_size):
            dis_target[idx, target_size - 1 - byte_idx] = tmp % 16
            tmp //= 16
    return binary, dis_target


def generate_code(passwd_length, batch_size, device, inv, use_minus_one, gen_random_WR):
    unsigned_long = np.random.randint(
        0, 2 ** passwd_length, size=(batch_size,), dtype=np.uint64)
    binary, dis_target = unsigned_long_to_binary_repr(unsigned_long, passwd_length)
    z = torch.from_numpy(binary).to(device)
    dis_target = torch.from_numpy(dis_target).to(device)

    repeated = True
    while repeated:
        rand_unsigned_long = np.random.randint(
            0, 2 ** passwd_length, size=(batch_size,), dtype=np.uint64)
        repeated = np.any(unsigned_long - rand_unsigned_long == 0)
    rand_binary, rand_dis_target = unsigned_long_to_binary_repr(
        rand_unsigned_long, passwd_length)
    rand_z = torch.from_numpy(rand_binary).to(device)
    rand_dis_target = torch.from_numpy(rand_dis_target).to(device)

    if not inv:
        if use_minus_one is True:
            z = z * 2 - 1
            rand_z = rand_z * 2 - 1
        elif use_minus_one == 'half':
            z = z * 2 - 1
        elif use_minus_one == 'one_fourth':
            z = z - 0.5
    return z, dis_target, rand_z, rand_dis_target
