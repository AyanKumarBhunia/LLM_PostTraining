# LLM_PostTraining

A tutorial-style Jekyll site for LLM post-training, alignment, and fine-tuning. It uses [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) and is set up for GitHub Pages.

## Quick Start

### 1. Push to GitHub

Create a new repository and push this project. For a project site, the expected deployment path is `username.github.io/LLM_PostTraining`. For a user site (`username.github.io`), set `baseurl` to `""`.

### 2. Configure `_config.yml`

Update these values before deploying:

- `url` — for example `https://username.github.io`
- `baseurl` — `"/LLM_PostTraining"` for a project site, or `""` for a user site
- `github.username`, `social.name`, `social.email`, and `social.links`

### 3. Enable GitHub Pages

1. Go to the repository's **Settings** -> **Pages**
2. Under **Build and deployment**, choose **GitHub Actions**
3. Push to `main` or `master` to trigger the site build

### 4. Local Preview

```bash
bundle install
bundle exec jekyll serve
```

Then open [http://127.0.0.1:4000/LLM_PostTraining/](http://127.0.0.1:4000/LLM_PostTraining/).

## Writing Posts

Add Markdown files to `_posts/` using the naming pattern `YYYY-MM-DD-title.md`. A simple starting front matter block looks like this:

```yaml
---
title: "Your Post Title"
date: 2026-06-06 12:00:00 +0000
categories: [fundamentals]
tags: [llm, post-training, sft]
math: false
pin: false
---
```

Suggested topic areas:

- LLM fundamentals and architecture
- pre-training vs post-training
- instruction tuning and SFT
- preference optimization (DPO, RLHF)
- data curation and formatting
- evaluation and benchmarking
- deployment and serving

## License

[MIT](LICENSE)
