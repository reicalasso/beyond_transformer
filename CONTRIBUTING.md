# Contributing to Beyond Transformer

First off, thanks for taking the time to contribute!

The following is a set of guidelines for contributing to Beyond Transformer. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Code of Conduct

This project and everyone participating in it is governed by the [Beyond Transformer Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for Beyond Transformer. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

**Before Submitting A Bug Report**

* **Check the [FAQ](docs/faq.md)** for a list of common questions and problems.
* **Determine which repository the problem should be reported in**.
* **Perform a [cursory search](https://github.com/yourusername/beyond_transformer/search?q=&type=Issues)** to see if the problem has already been reported. If it has **and the issue is still open**, add a comment to the existing issue instead of opening a new one.

**How Do I Submit A (Good) Bug Report?**

Bugs are tracked as [GitHub issues](https://guides.github.com/features/issues/). Create an issue on the repository and provide the following information by filling in [the template](.github/ISSUE_TEMPLATE/bug_report.md).

Explain the problem and include additional details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include links to files or GitHub projects, or copy/pasteable snippets, which you use in those examples.
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **Include screenshots and animated GIFs** which show you following the described steps and clearly demonstrate the problem.
* **If you're reporting that Beyond Transformer crashed**, include a crash report with a stack trace from the operating system.

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for Beyond Transformer, including completely new features and minor improvements to existing functionality.

**Before Submitting An Enhancement Suggestion**

* **Check the [FAQ](docs/faq.md)** for a list of common questions and problems.
* **Perform a [cursory search](https://github.com/yourusername/beyond_transformer/search?q=&type=Issues)** to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.

**How Do I Submit A (Good) Enhancement Suggestion?**

Enhancement suggestions are tracked as [GitHub issues](https://guides.github.com/features/issues/). Create an issue on the repository and provide the following information:

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Provide specific examples to demonstrate the steps**.
* **Describe the current behavior and explain which behavior you expected to see instead** and why.
* **Include screenshots and animated GIFs** which help you demonstrate the steps or point out the part of Beyond Transformer which the suggestion is related to.
* **Explain why this enhancement would be useful** to most Beyond Transformer users.

### Pull Requests

The process described here has several goals:

- Maintain Beyond Transformer's quality
- Fix problems that are important to users
- Engage the community in working toward the best possible Beyond Transformer
- Enable a sustainable system for Beyond Transformer's maintainers to review contributions

Please follow these steps to have your contribution considered by the maintainers:

1. Follow all instructions in [the template](.github/PULL_REQUEST_TEMPLATE.md)
2. Follow the [styleguides](#styleguides)
3. After you submit your pull request, verify that all [status checks](https://help.github.com/articles/about-status-checks/) are passing.

While the prerequisites above must be satisfied prior to having your pull request reviewed, the reviewer(s) may ask you to complete additional design work, tests, or other changes before your pull request can be ultimately accepted.

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
* When only changing documentation, include `[ci skip]` in the commit title

### Python Styleguide

All Python code must adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/).

### Documentation Styleguide

* Use [Markdown](https://daringfireball.net/projects/markdown/).
* Reference methods and classes in markdown with the custom `{}` notation:
    * Reference classes with `{ClassName}`
    * Reference instance methods with `{ClassName.method_name}`
    * Reference class methods with `{ClassName.class_method_name}`

## Additional Notes

### Issue and Pull Request Labels

This section lists the labels we use to help us track and manage issues and pull requests.

[GitHub search](https://help.github.com/articles/searching-issues/) makes it easy to use labels for finding groups of issues or pull requests you're interested in.

The labels are loosely grouped by their purpose, but it's not required that every issue have a label from every group or that an issue can't have more than one label from the same group.

#### Type of Issue and Issue State

* **bug** - Issues that are bugs.
* **enhancement** - Issues that are feature requests.
* **documentation** - Issues for improving documentation.
* **question** - Issues that are questions.
* **wontfix** - Issues that won't be fixed.
* **duplicate** - Issues that are duplicates of other issues.
* **help wanted** - Issues that need help from the community.
* **good first issue** - Issues that are good for new contributors.

#### Topic Categories

* **model** - Issues related to the model.
* **training** - Issues related to training.
* **data** - Issues related to data.
* **performance** - Issues related to performance.