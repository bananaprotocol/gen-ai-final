{
  description = "GenAI flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;

      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python3
          uv
        ];

        env = {
          LD_LIBRARY_PATH = lib.makeLibraryPath (pkgs.pythonManylinuxPackages.manylinux1 ++ [ pkgs.zstd ]);
        };

        shellHook = ''
          unset PYTHONPATH
          uv sync --extra dev
          . .venv/bin/activate
        '';
      };
    };
}
