module UtilsIO
using Flux, MLDatasets, BSON

function mnist_data()
    train = MLDatasets.MNIST(:train)
    test  = MLDatasets.MNIST(:test)

    Xtr = Float32.(reshape(train.features, :, 60000))
    Xtst= Float32.(reshape(test.features,  :, 10000))
    ytr = Flux.onehotbatch(train.targets, 0:9)
    ytst= Flux.onehotbatch(test.targets,   0:9)
    return Xtr, ytr, Xtst, ytst
end

save_obj(obj, name) = BSON.@save joinpath("models", "$(name).bson") obj

load_obj(name) = BSON.load(joinpath("models", "$(name).bson"))[:obj]
end # module
