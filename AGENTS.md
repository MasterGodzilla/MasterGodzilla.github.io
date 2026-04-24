# Repository Guidelines

## Project Structure & Module Organization

This repository is a Jekyll-powered academic website based on Academic Pages. Site configuration lives in `_config.yml`; navigation and shared data live in `_data/`. Content collections are organized by purpose: `_pages/`, `_posts/`, `_publications/`, `_portfolio/`, `_talks/`, `_teaching/`, and `_drafts/`. Layout templates are in `_layouts/`, reusable snippets are in `_includes/`, Sass partials are in `_sass/`, and compiled/static assets are under `assets/`, `images/`, and `files/`. Helper scripts and notebooks for generating Markdown are in `markdown_generator/`, `talkmap/`, `talkmap.ipynb`, and `talkmap.py`.

## Build, Test, and Development Commands

- `bundle install`: install Ruby and Jekyll dependencies from `Gemfile`.
- `jekyll serve -l -H localhost`: build the site and serve it locally at `http://localhost:4000` with live reload.
- `npm install`: install JavaScript build dependencies when editing `assets/js/`.
- `npm run build:js`: minify JavaScript into `assets/js/main.min.js`.
- `npm run watch:js`: rebuild minified JavaScript when source files change.
- `docker build -t jekyll-site .` and `docker run -p 4000:4000 --rm -v $(pwd):/usr/src/app jekyll-site`: run the site through Docker.

## Coding Style & Naming Conventions

Use Markdown with YAML front matter for site content. Match existing collection naming: dated posts use `YYYY-MM-DD-title.md`, while publication, talk, and page files should use short lowercase slugs. Keep front matter keys consistent with neighboring files. Use two-space indentation in YAML, HTML, and Sass. Edit Sass partials in `_sass/` and JavaScript sources in `assets/js/_main.js` or `assets/js/plugins/`; regenerate compiled files only when those sources change.

## Testing Guidelines

There is no dedicated automated test suite in this repository. Validate changes by running `jekyll serve -l -H localhost` and checking the affected pages for build warnings, broken layout, missing assets, and correct links. For JavaScript edits, run `npm run build:js` and verify the generated `assets/js/main.min.js` behaves as expected in the browser.

## Commit & Pull Request Guidelines

Recent commits use short, imperative or descriptive subjects such as `papers` and `removed teaching/talks`. Keep commit messages concise and focused on one change. Pull requests should describe the changed pages or assets, include screenshots for visual changes, mention any generated files, and link related issues when applicable. Confirm local Jekyll builds before requesting review.

## Agent-Specific Instructions

Keep edits scoped to the requested content or theme area. Do not rewrite vendored Sass, font files, or generated minified JavaScript unless the source change requires it. Preserve user content and avoid broad template refactors unless explicitly requested.
Treat original source drafts, including files under `blogs_drafts/`, as read-only unless the user explicitly asks to edit those originals; create derived posts or assets elsewhere instead of modifying source drafts in place.
