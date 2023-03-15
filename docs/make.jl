using Documenter
using jac

makedocs(
    sitename = "jac",
    format = Documenter.HTML(),
    modules = [jac],
    pages = [
        "Home" => "index.md",
        "Development Blogs" => Any[
            "Development/Scalar_AD.md",
        ],
        "Examples" => Any[
            "Examples/Simple_Scalar_Arithmetics.md",
        ],
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/tgautam03/jac.git"
)