# Running the Site Locally

This site requires **Ruby 3.1+**. Use whichever option is easiest on your machine.

## Option 1: Dev Container

1. Open the project in your editor
2. Reopen it in the provided dev container
3. Run `bash tools/run.sh`
4. Open `http://127.0.0.1:4000/LLM_PostTraining/`

## Option 2: Docker

```bash
cd /path/to/LLM_PostTraining
docker run --rm -v "$(pwd):/srv/jekyll" -p 4000:4000 jekyll/jekyll:4.2.2 jekyll serve -H 0.0.0.0 -P 4000 --livereload
```

Then open `http://127.0.0.1:4000/LLM_PostTraining/`.

## Option 3: Local Ruby via rbenv

```bash
brew install rbenv ruby-build
rbenv init
# Restart the shell, then:
rbenv install 3.3.0
rbenv local 3.3.0
cd /path/to/LLM_PostTraining
bundle install
bash tools/run.sh
```

Then open `http://127.0.0.1:4000/LLM_PostTraining/`.
