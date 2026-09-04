#!/usr/bin/env bash
# Grow swap to 16 GB on the NVMe so TensorRT engine builds fit in the Orin Nano's
# 8 GB *unified* memory. The first ORT TensorRT-EP engine build is the memory peak;
# with the stock 2 GB swap it OOMs. 16 GB swap (we have ~80 GB free on the NVMe)
# clears it. swappiness=10 keeps the kernel from paging unless it actually must.
#
# Run once:  sudo ./scripts/setup-swap.sh
set -euo pipefail

SWAPFILE="${1:-/swapfile}"
SIZE_GB="${2:-16}"

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 1
fi

echo ">> Disabling existing swap on ${SWAPFILE} (if any) ..."
swapoff "${SWAPFILE}" 2>/dev/null || true
rm -f "${SWAPFILE}"

echo ">> Allocating ${SIZE_GB}G at ${SWAPFILE} ..."
fallocate -l "${SIZE_GB}G" "${SWAPFILE}" || dd if=/dev/zero of="${SWAPFILE}" bs=1M count=$((SIZE_GB*1024)) status=progress
chmod 600 "${SWAPFILE}"
mkswap "${SWAPFILE}"
swapon "${SWAPFILE}"

echo ">> Persisting in /etc/fstab ..."
if ! grep -qE "^[^#]*[[:space:]]${SWAPFILE}[[:space:]]|^${SWAPFILE}[[:space:]]" /etc/fstab; then
  echo "${SWAPFILE} none swap sw 0 0" >> /etc/fstab
fi

echo ">> Setting vm.swappiness=10 ..."
if ! grep -q '^vm.swappiness' /etc/sysctl.conf; then
  echo 'vm.swappiness=10' >> /etc/sysctl.conf
fi
sysctl vm.swappiness=10 >/dev/null

echo
echo ">> Done. Current memory + swap:"
free -h
