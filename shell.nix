# Pinned development shell.
#
# `<nixpkgs>` is deliberately not used: it resolves through the *user's*
# channel or flake registry, so the Rust version would differ per machine and
# could shift under you after an unrelated `nix-channel --update`. The rev
# below is the nixpkgs this project has been built and tested against
# (nixpkgs-unstable as of 2026-09-09, rustc 1.97.1).
#
# To move it: pick a new rev, then
#   nix-prefetch-url --unpack https://github.com/NixOS/nixpkgs/archive/<rev>.tar.gz
# and update both fields. Overriding `pkgs` on the command line still works if
# you want to build against something else.
{
  pkgs ? import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/5545adfad2e98de106a5544ca7067e03010410bd.tar.gz";
    sha256 = "15drg799a8af6jki0077kfb3vi25ifyngcwffvf9sds02gmx1094";
  }) { },
}:

pkgs.mkShell {
  packages = with pkgs; [
    git
    cargo
    clippy
    rust-analyzer
    rustc
    rustfmt
  ];

  # rust-analyzer needs the stdlib sources; nixpkgs' rustc doesn't ship them.
  RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
}
