project = "humanoid-mrl"
copyright = "2026, MRL"
author = "Humanoid Team, MRL"

extensions = ["sphinx_copybutton"]
exclude_patterns = ["_build", "html", "Thumbs.db", ".DS_Store"]

html_theme = "shibuya"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"
html_theme_options = {
    "light_logo": "_static/logo_mrl_with_text_lightmode.svg",
    "dark_logo": "_static/logo_mrl_with_text_darkmode.svg",
    "accent_color": "indigo",
}
pygments_style = "sphinx"
