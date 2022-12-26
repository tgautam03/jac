using Documenter
using jac

makedocs(
    sitename = "jac",
    format = Documenter.HTML(),
    modules = [jac],
    pages = [
        "Home" => "index.md",
        "Development Blogs" => Any[
            "Development/AD_0.1.md",
        ],
        "Examples" => Any[
            "Examples/Intro_AD.md",
        ],
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/tgautam03/jac.git"
)
