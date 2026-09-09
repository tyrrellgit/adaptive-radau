{ pkgs ? import <nixpkgs> {} }:

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
