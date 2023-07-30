name = "Likelihood Options"
description = None

args_and_kwargs = (
    (("--studentt-likelihood-dof",), { 
        "help":"Degrees of freedom for student t likelihood function.",
        "type":float, 
        "metavar":'DOF', 
        "default":None,
    }),

    (("--refine-uncertainties",), { 
        "help":"Use Evans' 2011 error model from SCALA to correct uncertainties.",
        "action":'store_true', 
        "default":False,
    }),

    (("--multi-xtal-weighting",), {
        "help":"Learn per-crystal weights for the likelihood.",
        "action":'store_true',
        "default":False,
    }),

    (("--wc-regularize-weight",), {
        "help": "Set the weight of the term that regularizes crystal weights. "
                "By default, there is no regularization.",
        "type": float, 
        "default": 0.0,
    })
)

