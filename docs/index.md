# Quick-Tune-Tool

**A Practical Tool and User Guide for Automatically Finetuning Pretrained Models**

> Quick-Tune-Tool is an automated solution designed to streamline the process of selecting and finetuning pretrained models across various machine learning domains. Built upon the Quick-Tune algorithm, this tool abstracts complex research-level code into a user-friendly framework, making model finetuning accessible and efficient for practitioners.

---


## Installation
```bash
pip install quicktunetool
# or
git clone https://github.com/automl/quicktunetool
pip install -e quicktunetool  # Use -e for editable mode
```

---

## Usage

A simple example for using Quick-Tune-Tool with a pretrained optimizer for image classification:

```python
from qtt import QuickTuner, get_pretrained_optimizer
from qtt.finetune.cv.classification import finetune_script

# Load task information and meta-features
task_info, metafeat = extract_task_info_metafeat("path/to/dataset")

# Initialize the optimizer
optimizer = get_pretrained_optimizer("mtlbm/micro")
optimizer.setup(128, metafeat)

# Create QuickTuner instance and run
qt = QuickTuner(optimizer, finetune_script)
qt.run(task_info, time_budget=3600)
```

This code snippet demonstrates how to run QTT on an image dataset in just a few lines of code.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

For any questions or suggestions, please contact the maintainers.

---

## Project Status

- ✅ Active development

---

## Support

- 📝 [Documentation](https://automl.github.io/quicktunetool/)
- 🐛 [Issue Tracker](https://github.com/automl/quicktunetool/issues)
- 💬 [Discussions](https://github.com/automl/quicktunetool/discussions)

---

## License

This project is licensed under the BSD License - see the LICENSE file for details.

---

Made with ❤️ by https://github.com/automl