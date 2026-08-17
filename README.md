# Pranjal Gulati

Personal website and Writing archive, built with Jekyll and published at
[devpranjal.github.io](https://devpranjal.github.io).

## Structure

- `_data/` contains structured homepage content.
- `_layouts/` and `_includes/` contain the shared Jekyll presentation.
- `_posts/` contains Writing articles; `legacy/` preserves their previous URLs.
- `pages/` contains secondary pages, feeds, and compatibility redirects.
- `assets/` contains site media.
- `resume/` is ignored local source material and must not be published.
- `sysdesign/`, `upsc/`, `dsa/`, and `llm-inference/` are standalone applications with stable public URLs.

## Local development

Build the site with Docker:

```powershell
docker run --rm -v "${PWD}:/srv/jekyll" -w /srv/jekyll jekyll/jekyll:4 sh -lc "bundle install && bundle exec jekyll build"
```

Serve it at `http://127.0.0.1:4000/`:

```powershell
docker run --rm -p 4000:4000 -v "${PWD}:/srv/jekyll" -w /srv/jekyll jekyll/jekyll:4 sh -lc "bundle install && bundle exec jekyll serve --host 0.0.0.0 --port 4000"
```