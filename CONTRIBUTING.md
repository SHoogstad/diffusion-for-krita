# Contributing guide

Patches and contributions are very welcome!

## Reporting issues

Questions, help with setup, ideas and results go in [Discussions](https://github.com/SHoogstad/diffusion-for-krita/discussions). Issues are for bugs.

If you are facing an issue or want to report a bug, please check [existing issues](https://github.com/SHoogstad/diffusion-for-krita/issues).

Having a look at the log files can provide additional information for troubleshooting. You can find them in the `.logs` subfolder of the plugin installation folder (`ai_diffusion`). There is also a link in the plugin's connection settings.

When you open a new issue, please attach the log files. Other useful information to include: OS (Windows/Linux/Mac), Krita version, Plugin version, GPU vendor.

## Translations (Localization)

You can create or improve a translation for the plugin interface into your language.
Language files are stored in the `ai_diffusion/language` folder. Each translation has its own
file, typically named using a language code (eg. `en.json` for English).
If no file for your language exists, you can use the `new_language.json.template` file and
rename it. It will show up in the plugin's language settings after a restart.

You can check [existing translations here](https://github.com/SHoogstad/diffusion-for-krita/tree/main/ai_diffusion/language) - this might be more up-to-date than your local installation!

To edit a localization file, open the file in a text editor and provide translations for
each of the english text strings. For example, to provide a German translation, it could look
like this:
```json
{
  "id": "de",
  "name": "Deutsch",
  "translations": {
    "(Custom)": "(Benutzerdefiniert)",
    "<No text prompt>": "<Keine Text-Eingabe>",
    "Active": "Aktiv",
    "Add Content": "Inhalte hinzufügen",
    "Could not find LoRA '{lora}' used by sampler preset '{name}'": "LoRA '{lora}' konnte nicht gefunden werden, wird aber von Sampler '{name}' benutzt",
    ...
  }
}
```

**Important:** `{placeholders}` must be left unmodified! They will be replaced with actual content during runtime.

To update an existing translation (eg. after the plugin has been updated and new text was added)
simply search for entries which are `null`. These are valid, but not translated yet.

## Contributing code

For bigger changes, it makes sense to start an [idea discussion](https://github.com/SHoogstad/diffusion-for-krita/discussions/categories/ideas) or an issue first to discuss a proposal before time is committed.

You can submit your changes by opening a [pull request](https://github.com/SHoogstad/diffusion-for-krita/pulls).

### Branches

* `main` is the released code. Every commit on it must work, and each release is tagged on it.
  `main` is protected: no force pushes and no deleting it, so never rewrite its history.
* Do all work on a short-lived branch made from the latest `main`. Name it in lowercase
  kebab-case after what it does, not after who made it:
  * features: the feature or model, eg. `prompt-tools`, `p-h3-image`, `pose-match-proportions`
  * fixes: `fix-<what>`, eg. `fix-control-layer-crash`
  * docs or repo chores: `docs-<what>`, `chore-<what>`
* One topic per branch. Unrelated fixes go on their own branch, so they can be released alone.
* Changes reach `main` only through a pull request, merged with **Create a merge commit**
  (squash and rebase merging are turned off). A merge commit keeps the branch's own commits,
  so GitHub shows the branch as fully merged.
* If `main` moved on while the branch was open, update the branch with `git rebase main`
  before review. Do not merge `main` into your branch.
* The branch is deleted automatically when its pull request is merged. Only `main` lives on
  long-term.
* Experimental work that ships as a pre-release may stay on its branch until the pre-release
  is promoted to a full release; then it is merged into `main` through a pull request.

### Commits

* One logical change per commit. Formatting, renames and refactors go in separate commits
  from behaviour changes.
* Subject line: imperative mood, capitalized, no trailing period, at most 72 characters.
  Write what the change does for the user, not which files were touched.
* Prefix the subject with the area when it helps: a model, workspace or component followed by
  a colon, eg. `MOSS-Image: keep the first region's box ...`, `Pose: fix skeleton editing ...`,
  `Live: preview scales with the docker`, `Runpod: ...`.
* Leave an empty line after the subject. Use the body to explain *why* and anything a reviewer
  can't see from the diff (tested hardware, ComfyUI version, known limits). Wrap at 72 characters.
* Reference issues in the body: `Fixes #123` or `See #123`.
* Every commit must pass `ruff format`, `ruff check`, `pyright` and `pytest tests --ci`.
  CI does not run on push, so run them locally.
* Commit with your GitHub noreply address (`<id>+<user>@users.noreply.github.com`) if you
  don't want your private email in public history.
* Never commit secrets (API keys, tokens), model files, logs or build output (`*.zip`).

### Releases

* Version scheme: `<generic>.<feature>.<fix>`. Big features bump the middle number, everything
  else bumps the last, eg. `1.60.1` → `1.61.0` (feature) or `1.61.1` (fix).
* The release commit only bumps `__version__` in `ai_diffusion/__init__.py` and has the
  subject `Version X.Y.Z` (optionally `Version X.Y.Z: <one-line summary>`).
* Tag that commit `vX.Y.Z` and attach `diffusion_for_krita-X.Y.Z.zip`. The auto-updater reads
  the latest GitHub release, so a release must have the zip and be marked latest
  (or pre-release for testing builds).
* Never reuse a version number. If a release is pulled, the next one gets a higher number,
  so installs that already got the pulled version still update.

### Plugin development

The easiest way to run a development version of the plugin is to create symlinks to the git repository where Krita expects to find Python plugins:
1. `git clone` the repository into a location of your choice
1. `git submodule update --init`
1. `./install.py`

### Code formatting

The codebase uses [ruff](https://docs.astral.sh/ruff/) for linting. You can
use an IDE integration, or check locally by running in the repository root:
```
ruff format
ruff check
```

### Code style

Code style follows the official Python recommendations. Only exception: no `ALL_CAPS`.

### Type checking

Type annotations should be used where types can't be inferred. Basic type checks are enabled for the project and should not report errors.

The `Krita` module is special in that it is usually only available when running inside Krita. To make type checking work an interface file is located in `scripts/typeshed`.

You can run `pyright` from the repository root to perform type checks on the entire codebase. Please do this before opening a pull request.

### Debug

You can attach a debugger to a running Krita process. This allows to use the visual
debugger for exceptions, breakpoints, inspecting and stepping through the code.

The way it works is:
1. `debugpy` is added to the `ai_diffusion` folder as a git submodule to make it
   available inside Krita's embedded Python
1. `extension.py` starts a debug server on port 5678 if the `debugpy` module is present
   (skipped for release deployments)
2. VSCode (or more generally any `debugpy` client) attaches to the server

For VSCode, add this configuration to your own `.vscode/launch.json` and start it via the
"Run and Debug" tab (F5). On Windows, use `${env:APPDATA}/krita/pykrita` as `remoteRoot`.
```json
{
  "name": "Attach to Krita",
  "type": "debugpy",
  "request": "attach",
  "connect": { "host": "localhost", "port": 5678 },
  "pathMappings": [
    {
      "localRoot": "${workspaceFolder}",
      "remoteRoot": "${env:HOME}/.local/share/krita/pykrita"
    }
  ]
}
```

You can also add breakpoints inside the code with `import debugpy; debugpy.breakpoint()`.

### Tests

There are tests, although with some caveats currently.

To install dependencies for tests run:
```
pip install -r requirements.txt
```
Tests are run from the project root via pytest:
```
pytest tests
```
Some tests require a running ComfyUI server. This should be automated... but for now it's not.

### What is tested
Generating images is tested. Because it takes a lot of time the number of tests is limited. Images are not compared in most cases, as they tend to frequently change with updates to dependencies.

Functionality which uses Krita's API is _not_ tested. It just doesn't work outside Krita without a comprehensive mock.

UI is _not_ tested. Because UI.

Everything else has tests. Mostly. If effort is reasonable, tests are expected. They help being confident about making changes.

### Testing the installer

Testing changes to the installer is annoying because of the file sizes involved. There are some things that help. You can preload model files with the following script:
```
python scripts/download_models.py --minimal scripts/downloads
```
This will download the minimum required models and store them in `scripts/downloads`.

The following command does some automated testing for installation and upgrade. It starts a local file server which pulls preloaded models, so it's reasonably fast and doesn't download the entire internet.
```
pytest tests/test_server.py --test-install -vs
```
You can also run the file server manually. Then you can start Krita with the `HOSTMAP` environment variable set, and it will map HuggingFace & civit.ai links to localhost.
```
python scripts/file_server.py

HOSTMAP=1 /your/krita/install/krita
