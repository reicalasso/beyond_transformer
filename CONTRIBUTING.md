# Contributing to Beyond Transformer

First off, thank you for considering contributing to Beyond Transformer! It's people like you that make this project such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [maintainer-email@example.com].

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

Before creating bug reports, please check [this list](#before-submitting-a-bug-report) as you might find out that you don't need to create one. When you are creating a bug report, please [include as many details as possible](#how-do-i-submit-a-good-bug-report). Fill out [the required template](https://github.com/yourusername/beyond_transformer/.github/ISSUE_TEMPLATE/bug_report.md), the information it asks for helps us resolve issues faster.

> **Note:** If you find a **Closed** issue that seems like it is the same thing that you're experiencing, open a new issue and include a link to the original issue in the body of your new one.

#### Before Submitting A Bug Report

* **Check the [documentation](docs/)** for tips on troubleshooting.
* **Perform a cursory search** to see if the problem has already been reported. If it has **and the issue is still open**, add a comment to the existing issue instead of opening a new one.

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitHub issues](https://guides.github.com/features/issues/). Create an issue and provide the following information by filling in [the template](https://github.com/yourusername/beyond_transformer/.github/ISSUE_TEMPLATE/bug_report.md).

Explain the problem and include additional details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include links to files or GitHub projects, or copy/pasteable snippets, which you use in those examples.
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **Include screenshots and animated GIFs** which show you following the described steps and clearly demonstrate the problem.
* **If you're reporting that Beyond Transformer crashed**, include a crash report with a stack trace from the operating system.

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion, including completely new features and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your suggestion and find related suggestions.

Before creating enhancement suggestions, please check [this list](#before-submitting-an-enhancement-suggestion) as you might find out that you don't need to create one. When you are creating an enhancement suggestion, please [include as many details as possible](#how-do-i-submit-a-good-enhancement-suggestion). Fill in [the template](https://github.com/yourusername/beyond_transformer/.github/ISSUE_TEMPLATE/feature_request.md), including the steps that you imagine you would take if the feature you're requesting existed.

#### Before Submitting An Enhancement Suggestion

* **Check the [documentation](docs/)** for tips on existing features.
* **Perform a cursory search** to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.

#### How Do I Submit A (Good) Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://guides.github.com/features/issues/). Create an issue and provide the following information:

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include copy/pasteable snippets which you use in those examples, as [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
* **Include screenshots and animated GIFs** which help you demonstrate the steps or point out the part of Beyond Transformer which the suggestion is related to.
* **Explain why this enhancement would be useful** to most Beyond Transformer users.

### Your First Code Contribution

Unsure where to begin contributing to Beyond Transformer? You can start by looking through these `beginner` and `help-wanted` issues:

* [Beginner issues][beginner] - issues which should only require a few lines of code, and a test or two.
* [Help wanted issues][help-wanted] - issues which should be a bit more involved than `beginner` issues.

Both issue lists are sorted by total number of comments. While not perfect, number of comments is a reasonable proxy for impact a given change will have.

### Pull Requests

The process described here has several goals:

- Maintain Beyond Transformer's quality
- Fix problems that are important to users
- Engage the community in working toward the best possible Beyond Transformer
- Enable a sustainable system for Beyond Transformer's maintainers to review contributions

Please follow these steps to have your contribution considered by the maintainers:

1. Follow all instructions in [the template](.github/pull_request_template.md)
2. Follow the [styleguides](#styleguides)
3. After you submit your pull request, verify that all [status checks](https://help.github.com/articles/about-status-checks/) are passing

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

All Python code should be formatted with [Black](https://github.com/ambv/black) and [isort](https://github.com/timothycrosley/isort).

### Documentation Styleguide

* Use [Markdown](https://daringfireball.net/projects/markdown).
* Reference methods and classes in documentation using backticks.
* When referencing parameters, wrap them in asterisks.

## Additional Notes

### Issue and Pull Request Labels

This section lists the labels we use to help us track and manage issues and pull requests.

[GitHub search](https://help.github.com/articles/searching-issues/) makes it easy to use labels for finding groups of issues or pull requests you're interested in.

The labels are loosely grouped by their purpose, but it's not required that every issue have a label from every group or that an issue can't have more than one label from the same group.

#### Type of Issue and Issue State

| Label name | Description |
| --- | --- |
| `bug` | Confirmed bugs or reports that are very likely to be bugs |
| `enhancement` | Feature requests |
| `question` | Questions more than bug reports or feature requests |
| `documentation` | Issues for improving documentation |
| `beginner` | Good for newcomers |
| `help-wanted` | Help wanted on this issue |
| `duplicate` | This issue or pull request already exists |
| `invalid` | This doesn't seem right |
| `wontfix` | This will not be worked on |

#### Topic Categories

| Label name | Description |
| --- | --- |
| `model` | Related to model architecture |
| `training` | Related to training process |
| `performance` | Related to performance optimization |
| `memory` | Related to memory management |
| `attention` | Related to attention mechanisms |
| `state-management` | Related to state management |
| `debugging` | Related to debugging tools |
| `testing` | Related to testing infrastructure |
| `visualization` | Related to visualization tools |

## Community

Discussions about Beyond Transformer take place on this repository's [Issues](https://github.com/yourusername/beyond_transformer/issues) and [Pull Requests](https://github.com/yourusername/beyond_transformer/pulls) sections. Anybody is welcome to join these conversations.

Wherever possible, we encourage public communication so that anyone can chime in and benefit from the discussion.

## Recognition

We deeply appreciate contributions to Beyond Transformer. Contributors are recognized in several ways:

1. **GitHub Recognition**: All contributors appear in our [contributors list](https://github.com/yourusername/beyond_transformer/graphs/contributors)
2. **Release Notes**: Significant contributions are highlighted in our release notes
3. **Acknowledgments**: Outstanding contributors may be acknowledged in publications or presentations

Thank you for helping make Beyond Transformer a better project for everyone!