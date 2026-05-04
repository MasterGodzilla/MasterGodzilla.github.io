# Repository Guidelines

## Project Structure & Module Organization

This repository is a Jekyll-powered academic website based on Academic Pages. Site configuration lives in `_config.yml`; navigation and shared data live in `_data/`. Content collections are organized by purpose: `_pages/`, `_posts/`, `_publications/`, `_portfolio/`, `_talks/`, `_teaching/`, and `_drafts/`. Layout templates are in `_layouts/`, reusable snippets are in `_includes/`, Sass partials are in `_sass/`, and compiled/static assets are under `assets/`, `images/`, and `files/`. Helper scripts and notebooks for generating Markdown are in `markdown_generator/`, `talkmap/`, `talkmap.ipynb`, and `talkmap.py`.

## Coding Style & Naming Conventions

Use Markdown with YAML front matter for site content. Match existing collection naming: dated posts use `YYYY-MM-DD-title.md`, while publication, talk, and page files should use short lowercase slugs. Keep front matter keys consistent with neighboring files. Use two-space indentation in YAML, HTML, and Sass. Edit Sass partials in `_sass/` and JavaScript sources in `assets/js/_main.js` or `assets/js/plugins/`; regenerate compiled files only when those sources change.

## Testing Guidelines

There is no dedicated automated test suite in this repository. For agent work, do not run local build, test, serve, Docker, npm build, or browser verification commands unless the user explicitly asks. GitHub Pages will build the site after push.

## Blog Workflow

Original writing drafts may live in `blogs_drafts/` or `_drafts/`. Treat those source drafts as read-only unless the user explicitly asks to edit them. To publish a blog post, create or update a derived Markdown file in `_posts/` using the dated filename format `YYYY-MM-DD-title-slug.md`.

Published posts need YAML front matter consistent with neighboring files in `_posts/`, usually including `title`, `date`, `permalink`, `tags`, and related metadata when appropriate. Copy or adapt the draft content into the `_posts/` file, clean up formatting for Jekyll, update image links to point at committed assets under `images/` or another stable site path, and leave the original draft intact.

## Commit & Pull Request Guidelines

Recent commits use short, imperative or descriptive subjects such as `papers` and `removed teaching/talks`. Keep commit messages concise and focused on one change. Pull requests should describe the changed pages or assets, include screenshots for visual changes, mention any generated files, and link related issues when applicable.

## Agent-Specific Instructions

Keep edits scoped to the requested content or theme area. Do not rewrite vendored Sass, font files, or generated minified JavaScript unless the source change requires it. Preserve user content and avoid broad template refactors unless explicitly requested.
Treat original source drafts, including files under `blogs_drafts/`, as read-only unless the user explicitly asks to edit those originals; create derived posts or assets elsewhere instead of modifying source drafts in place.
