{-----------------------------------------------------------------------
Artificial Neural Network in Haskell (Batch processing)
(c) 2017 Fraunhofer-Chalmers Research Centre for Industrial
         Mathematics, Department of Systems and Data Analysis

Research and development:
Gregor Ulm      - gregor.ulm@fcc.chalmers.se
Emil Gustavsson - emil.gustavsson@fcc.chalmers.se

In the hardcoded example in 'main' an artificial neural network (ANN)
with two input neurons, two hidden neurons, and three output neurons
is defined.

There are several hard-coded input values.

Note: with small values for 'eps' this program may run out of memory. We
did not investigate this issue further.
-----------------------------------------------------------------------}

type Col     = [Double]
type Weights = ([Col], [Col], [Col])
type Config  = (Col, Weights, Col)
type Configs = ([Col], Weights, [Col])
type Input   = [Col]


-- example objective function
sigmoid :: Double -> Double
sigmoid x = 1.0 / (1.0 + exp (-x))


-- multiply list of inputs with list of lists of weights
forward :: Col -> [Col] -> Col -> Col
forward _     []     acc = reverse acc
forward input (w:ws) acc = forward input ws (val:acc)
  where val = sum $ zipWith (*) input w


-- compute error in output layer
output_error :: Col -> Col -> Col
output_error = zipWith (\x y -> x * (1.0 - x) * (y - x))


-- Compute new weights of output layer
-- args: input values, errors of output layer, weights, accumulator
backprop :: Col -> Col -> [Col] -> [Col]-> [Col]
backprop _     []     []       acc = reverse acc
backprop input (e:es) (ws:wss) acc = backprop input es wss acc'
  where acc' = zipWith (\w i -> w + (e * i)) ws input : acc


-- compute errors of hidden layer neurons
-- args: hidden layer output, output errors, output layer weights, acc
err_hidden :: Col -> Col -> [Col] -> Col -> Col
err_hidden []     _       _  acc = reverse acc
err_hidden (h:hs) out_err ws acc = err_hidden hs out_err ws' acc'
  where
  -- take first element of output weights for backpropagation,
  -- results in a list of weights of outgoing edges for each hidden
  -- layer neuron
  out  = map head ws
  -- remaining weights for next iteration
  ws'  = map tail ws
  -- compute error for current hidden layer neuron
  acc' = sum (zipWith (*) out out_err) * h * (1.0 - h) : acc


-- Forward Pass
compute_forward :: Col -> Weights -> Col -> (Col, Col)
compute_forward input weights targets = (hidden_out, output_errors)
  where
  (w_input, w_bias, w_hidden) = weights
  hidden_in     = forward input w_input []

  -- % add bias; w_bias is a list of lists with one element each
  hidden_in'    = zipWith (\x y -> x + head y) hidden_in w_bias

  hidden_out    = map sigmoid hidden_in'
  output_in     = forward hidden_out w_hidden []
  output_out    = map sigmoid output_in
  output_errors = output_error output_out targets


ann :: Col -> Weights -> Col -> (Col, Weights)
ann input ws targets = (output_errors, (w_input', w_bias', w_hidden'))
  where
  (w_input, w_bias, w_hidden) = ws
  -- Forward Pass
  (hidden_out, output_errors) = compute_forward input ws targets
  -- Update weights for output layer
  w_hidden'     = backprop   hidden_out output_errors w_hidden  []
  hidden_errors = err_hidden hidden_out output_errors w_hidden' []
  w_input'      = backprop input hidden_errors w_input []
  w_bias'       = backprop [1]   hidden_errors w_bias  []


-- keep weights, iterate over input/target pairs
wrap_ann []     ws []     errors = (reverse errors, ws)
wrap_ann (i:is) ws (t:ts) errors = wrap_ann is ws' ts (error:errors)
  where (error, ws') = ann i ws t


iterate_eps :: Double -> Configs -> Int -> IO ()
iterate_eps eps (inputs, ws, targets) n = do
  let (errors, ws') = wrap_ann inputs ws targets []
      done          = check_xss errors eps

  if not done
  then
    if n `mod` 10000 == 0
    then do putStrLn $ "Errors in output layer: " ++ (show errors)
            putStrLn $ "End of iteration "        ++ (show n)
            iterate_eps eps (inputs, ws', targets) (n + 1)
    else iterate_eps eps (inputs, ws', targets) (n + 1)

  else do putStrLn   "-------------------------------------------------"
          putStrLn $ "Final errors in output layer: " ++ (show errors)
          putStrLn $ "Total iterations: "             ++ (show n)


-- input: list of lists of errors; checks if error for all inputs is
-- below threshold
check_xss []     _   = True;
check_xss (es:ess) eps =
    if all (\x -> abs x < eps) es
    then check_xss ess eps
    else False


main :: IO ()
main = iterate_eps 0.07 config 0
  where
  -- Hardcoded input: 2 input, 3 hidden, 2 output neurons
  inputs    = [[0.4, 0.4], [0.3, 0.2], [0.2, 0.2],
               [0.3, 0.3], [0.2, 0.1]]

  -- Weights from 2 input to 3 hidden nodes
  w_input  = [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]

  -- Weight from bias node (connects with hidden layer)
  w_bias = [[0.1], [0.1], [0.1]]

  -- Weights from 3 hidden to 2 output nodes
  w_hidden = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]

  targets  = [[0.0, 0.8], [0.1, 0.5], [0.0, 0.4],
              [0.0, 0.6], [0.1, 0.3]]
  -- Of course the numbers are arbitrary

  weights  = (w_input, w_bias, w_hidden)
  config   = (inputs, weights, targets)
