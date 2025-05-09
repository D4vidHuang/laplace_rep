using Flux, Statistics, Random, ArgParse, MLDatasets, LaplaceRedux
include("utils_io.jl"); using .UtilsIO

function accuracy(probs, labels)
    preds = map(i -> argmax(view(probs,:,i)) - 1, 1:size(probs,2))
    return mean(preds .== labels)
end

function eval_model(mode="map")
    X_tr, y_tr, X_te, y_te = UtilsIO.mnist_data()
    test_labels = vec(MLDatasets.MNIST(:test).targets)

    if mode == "map"
        model = UtilsIO.load_obj("map")
        probs = softmax(model(X_te))
    else           
        la = UtilsIO.load_obj(mode)
        nsamp = 20 
        probs = zeros(Float32, 10, size(X_te,2))
        println("Running Monte-Carlo sampling with $nsamp samples...")
        for s in 1:nsamp
            print("\rSample $s/$nsamp...")
            probs .+= predict(la, X_te) 
        end
        println("\nAveraging predictions...")
        probs ./= nsamp
    end
    println("[$(uppercase(mode))] Test accuracy = $(accuracy(probs, test_labels)*100) %")
end

# julia evaluate_la.jl --mode=la_star
if abspath(PROGRAM_FILE) == @__FILE__
    s = ArgParseSettings()
    @add_arg_table s begin
        "--mode"
            help = "map / la / la_star"
            arg_type = String
            default = "map"
    end
    args = parse_args(s)
    eval_model(args["mode"])
end
