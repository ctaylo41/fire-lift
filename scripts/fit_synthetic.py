"""Run the core synthetic inverse-rendering experiments.

Suggested CLI/config dimensions:
- representation: dense | axisymmetric
- views: 1 | 2 | 4 | 8 | 16
- lambda_tv
- lambda_sparse
- seed

Outputs to save:
- fitted volume/profile parameters
- loss history
- rendered reconstructions
- image error
- normalized GT volume error
"""


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
