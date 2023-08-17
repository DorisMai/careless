name = "Likelihood Options"
description = None

args_and_kwargs = (
    (("--studentt-likelihood-dof",), { 
        "help":"Degrees of freedom for student t likelihood function.",
        "type":float, 
        "metavar":'DOF', 
        "default":None,
    }),

    (("--xtal-weights-file",), {
        "help": "Initialize the raw crystal weights for likelihood from the ouput of a previous run. This argument should"
                "be a string beginning with the base filename used in the previous run and ending in _xtal_weights.  For "
                "instance, if the previous run was called with `careless mono [...] merge/hewl`, the appropriate file name"
                "would be merge/hewl_xtal_weights",
        "type": str,
        "default": None,
    }),

    (("--freeze-xtal-wc",), {
        "help": "Do not optimize the crystal weights in likelihood.",
        "action": "store_true"
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

