{
  description = "A Python development environment with TeX support";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        pythonPackages = with pkgs.python3Packages; [
          jupyter-core
          notebook
          ipykernel
          ipywidgets
          numpy
          scipy
          pandas
          scikit-learn
          matplotlib
          seaborn
          debugpy
        ];

        # TeX packages from default.nix
        tex = pkgs.texlive.combine {
          inherit
            (pkgs.texlive)
            scheme-minimal
            metapost
            xetex
            dvisvgm
            dvipng
            wrapfig
            amsmath
            ulem
            hyperref
            capt-of
            physics
            siunitx
            booktabs
            ;
        };

        devTools = with pkgs; [
          basedpyright
          ruff
          # TeX tools from default.nix
          tex
          texlab
          tectonic
          ghostscript
          imagemagick
        ];
      in {
        devShells.default = pkgs.mkShell {
          buildInputs =
            devTools
            ++ pythonPackages
            ++ [pkgs.python3];
        };

        packages = {
          default = pkgs.python3.withPackages (ps: pythonPackages);
          python = pkgs.python3;
        };

        formatter = pkgs.alejandra;
      }
    );
}
