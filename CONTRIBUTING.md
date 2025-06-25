# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/Priesemann-Group/mrestimator/issues.

If you are reporting a bug, please include ideally a minimal reproducible example with:
- Your operating system name and version
- Python version and mrestimator version
- Any details about your local setup that might be helpful in troubleshooting
- Detailed steps to reproduce the bug

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and
"help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with
"enhancement" and "help wanted" is open to whoever wants to implement
it.

### Write Documentation

Mr. Estimator could always use more documentation,
whether as part of the official docs,
in docstrings, or even on the web in blog posts, articles, and such.

If you're interested in improving documentation, you can:
- Fix typos and improve clarity in existing documentation
- Add examples and tutorials
- Improve docstrings in the code
- Create blog posts or tutorials about using mrestimator

### Submit Feedback

The best way to send feedback is to file an issue at
https://github.com/Priesemann-Group/mrestimator/issues.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to
  implement.
- Remember that this is a volunteer-driven project, and that
  contributions are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `mrestimator` for local development.

1. Fork the `mrestimator` repo on GitHub.

2. Clone your fork locally:

```bash
git clone git@github.com:your_name_here/mrestimator.git
```

3. Create a virtual environment with your favorite tool. We recommend using conda:

```bash
conda create --name mrestimator-dev python=3.11
conda activate mrestimator-dev
```

Or with venv:

```bash
python -m venv mrestimator-dev
source mrestimator-dev/bin/activate  # On Windows: mrestimator-dev\Scripts\activate
```

4. Install the package including dev dependencies in editable mode:

```bash
cd mrestimator/
pip install -e ".[dev]"
```

This will install:
- The mrestimator package in development mode
- All runtime dependencies (numpy, scipy, matplotlib)
- Optional dependencies (numba, tqdm)
- Development tools (pre-commit, ruff, testing tools)
- Documentation dependencies (sphinx, sphinx-book-theme, etc.)

5. Install pre-commit hooks:

```bash
pre-commit install
```

6. Create a branch for local development:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

Now you can make your changes locally.

7. When you're done making changes, check that your code passes the linter and formatter,
   the tests pass, and the documentation builds correctly:

```bash
# Run linting and formatting
make lint
make format

# Run tests
make test

# Build documentation
make docs-build

# Or preview documentation locally
make docs-preview
```

Look inside the `Makefile` for more commands. For instance:
- `make clean` - Remove build artifacts
- `make check` - Run both linting and tests
- `make all` - Run the full development cycle

8. Commit your changes and push your branch to GitHub:

```bash
git add .
git commit -m "Your detailed description of your changes."
git push origin name-of-your-bugfix-or-feature
```

Please write clear, descriptive commit messages that explain what changes you made and why.

9. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests for any new functionality.
2. If the pull request adds functionality, the docs should be updated.
   Put your new functionality into a function with a docstring.
3. The pull request should work for Python 3.10, 3.11, and 3.12.
4. Make sure all existing tests still pass.

## Coding Standards

### Code Style

We use `ruff` for code formatting and linting. The configuration is in `pyproject.toml`.
Run `make format` to automatically format your code and `make lint` to check for issues.

### Documentation Style

- Use NumPy-style docstrings for functions and classes
- Include examples in docstrings where helpful
- Keep line length to 88 characters (enforced by ruff)

### Testing

- Write tests for new functionality
- Use pytest for testing
- Place tests in the `tests/` directory
- Test file names should start with `test_`

## Development Environment

### Dependencies

The development environment includes:

**Core dependencies:**
- numpy (>=1.11)
- scipy (>=1.0.0)
- matplotlib (>=1.7.0)

**Optional runtime dependencies:**
- numba (>=0.44) - for performance improvements
- tqdm - for progress bars

**Development tools:**
- pre-commit - for git hooks
- ruff - for linting and formatting
- pytest - for testing

**Documentation tools:**
- sphinx - documentation generator
- sphinx-book-theme - modern documentation theme
- myst-parser - Markdown support in docs
- nbsphinx - Jupyter notebook support

### Performance Considerations

When running on clusters or in distributed environments, you may need to control
thread usage. Set these environment variables:

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
```

## Questions?

If you have questions about contributing, feel free to:
- Open an issue on GitHub
- Contact the maintainers directly
- Join discussions in existing issues

Thank you for contributing to Mr. Estimator! 