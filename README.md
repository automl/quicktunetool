# Quick-Tune-Tool

[![image](https://img.shields.io/pypi/l/quicktunetool.svg)](https://pypi.python.org/pypi/quicktunetool)
[![image](https://img.shields.io/pypi/pyversions/quikctunetool.svg)](https://pypi.python.org/pypi/quicktunetool)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**A Practical Tool and User Guide for Automatically Finetuning Pretrained Models**

> Quick-Tune-Tool is an automated solution for selecting and finetuning pretrained models across various machine learning domains. Built upon the Quick-Tune algorithm, this tool bridges the gap between research-code and practical applications, making model finetuning accessible and efficient for practitioners.


## Installation
```bash
pip install quicktunetool
# or
git clone https://github.com/automl/quicktunetool
pip install -e quicktunetool  # Use -e for editable mode
```


## Usage

A simple example for using Quick-Tune-Tool with a pretrained optimizer for image classification:

```python
from qtt import QuickTuner, get_pretrained_optimizer
from qtt.finetune.image.classification import fn

# Load task information and meta-features
task_info, metafeat = extract_task_info_metafeat("path/to/dataset")

# Initialize the optimizer
optimizer = get_pretrained_optimizer("mtlbm/micro")
optimizer.setup(128, metafeat)

# Create QuickTuner instance and run
qt = QuickTuner(optimizer, fn)
qt.run(task_info, time_budget=3600)
```

This code snippet demonstrates how to run QTT on an image dataset in just a few lines of code.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

For any questions or suggestions, please contact the maintainers.

## Project Status

- ✅ Active development

## Support

- 📝 [Documentation](https://automl.github.io/quicktunetool/)
- 🐛 [Issue Tracker](https://github.com/automl/quicktunetool/issues)
- 💬 [Discussions](https://github.com/automl/quicktunetool/discussions)

## License

This project is licensed under the BSD License - see the LICENSE file for details.

## References

The concepts and methodologies of QuickTuneTool are detailed in the following workshop paper:

```
@inproceedings{
rapant2024quicktunetool,
title={Quick-Tune-Tool: A Practical Tool and its User Guide for Automatically Finetuning Pretrained Models},
author={Ivo Rapant and Lennart Purucker and Fabio Ferreira and Sebastian Pineda Arango and Arlind Kadra and Josif Grabocka and Frank Hutter},
booktitle={AutoML Conference 2024 (Workshop Track)},
year={2024},
url={https://openreview.net/forum?id=d0Hapti3Uc}
}
```

If you use QuickTuneTool in your research, please also cite the following paper:

```
@inproceedings{
arango2024quicktune,
title={Quick-Tune: Quickly Learning Which Pretrained Model to Finetune and How},
author={Sebastian Pineda Arango and Fabio Ferreira and Arlind Kadra and Frank Hutter and Josif Grabocka},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=tqh1zdXIra}
}
```

---

Made with ❤️ by https://github.com/automl