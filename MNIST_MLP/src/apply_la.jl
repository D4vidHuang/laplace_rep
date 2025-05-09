using Flux, LaplaceRedux, MLUtils, ArgParse, ProgressMeter
include("utils_io.jl"); using .UtilsIO
using LaplaceRedux: KronHessian, FullHessian 

function build_dataset(X, Y)
    return [(X[:,i], Y[:,i]) for i in 1:size(X,2)]
end

function apply_la(; model_name="mlp", la_type="la")
    println("Loading MAP model...")
    model = UtilsIO.load_obj("map")  
    
    println("Loading MNIST data...")
    X_tr, y_tr, _, _ = UtilsIO.mnist_data()
    data = build_dataset(X_tr, y_tr)

    println("Setting up Laplace approximation...")
    hess = la_type == "la"      ? KronHessian() :
            la_type == "la_star" ? FullHessian() :
            error("la_type 必须是 \"la\" 或 \"la_star\"")

    la = Laplace(model;
                 likelihood=:classification,
                 subset_of_weights=:last_layer, 
                 hessian_structure=hess) 
    
    println("Fitting Laplace approximation...")
    fit!(la, data)
    
    println("Optimizing prior precision...")
    optimize_prior!(la; verbosity=1, n_steps=100)                  
    
    println("Saving Laplace model...")
    UtilsIO.save_obj(la, la_type)
    println("Done!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    using ArgParse
    s = ArgParseSettings()
    @add_arg_table s begin
        "--la_type"
            help = "la 或 la_star"
            arg_type = String
            default = "la"
    end
    args = parse_args(s)
    apply_la(; la_type=args["la_type"])
end
