using Flux, Random, Statistics
include("utils_io.jl"); using .UtilsIO

function train_map()
    Random.seed!(42)
    X_tr, y_tr, X_te, y_te = UtilsIO.mnist_data()

    model = Chain(
        Dense(784 => 128, relu),
        Dense(128 => 64, relu),
        Dense(64 => 10)
    )
    loss(x, y) = Flux.logitcrossentropy(model(x), y)

    opt = Flux.Adam(0.001)
    ps = Flux.params(model)


    batch_size = 128
    num_epochs = 10


    train_data = Flux.DataLoader((X_tr, y_tr), batchsize=batch_size, shuffle=true)
    test_data = Flux.DataLoader((X_te, y_te), batchsize=1000)

    for epoch in 1:num_epochs

        total_loss = 0.0
        num_batches = 0
        
        for (x, y) in train_data
  
            gs = gradient(() -> loss(x, y), ps)
            Flux.Optimise.update!(opt, ps, gs)

            total_loss += loss(x, y)
            num_batches += 1
        end

        train_loss = total_loss / num_batches
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        for (x, y) in test_data
            ŷ = model(x)
            test_loss += loss(x, y)
            correct += sum(Flux.onecold(ŷ) .== Flux.onecold(y))
            total += size(x, 2)
        end
        
        test_loss = test_loss / length(test_data)
        accuracy = correct / total
        
        @info "Epoch $epoch stats" train_loss=train_loss test_loss=test_loss accuracy=accuracy
    end

    UtilsIO.save_obj(model, "map")
    return model
end

if abspath(PROGRAM_FILE) == @__FILE__
    train_map()
end
