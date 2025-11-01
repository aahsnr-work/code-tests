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

        # Build Python with PGO optimizations enabled and LTO (already enabled by default on 64-bit systems)
        pythonInterpreter = let
          self = pkgs.python313.override {
            enableOptimizations = true;
            inherit self;
          };
        in
          self;

        pythonPackages = with pythonInterpreter.pkgs; [
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
            ++ [pythonInterpreter];
        };

        packages = {
          default = pythonInterpreter.withPackages (ps: pythonPackages);
          python = pythonInterpreter;
        };

        formatter = pkgs.alejandra;
      }
    );
}
