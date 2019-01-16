{-----------------------------------------------------------------------
Artificial Neural Network in Haskell (Basic Example)
(c) 2017 Fraunhofer-Chalmers Research Centre for Industrial
         Mathematics, Department of Systems and Data Analysis

Research and development:
Gregor Ulm      - gregor.ulm@fcc.chalmers.se
Emil Gustavsson - emil.gustavsson@fcc.chalmers.se

In the hardcoded example in 'main' an artificial neural network (ANN)
with two input neurons, two hidden neurons, and three output neurons
is defined.

There is only one hard-coded input value.
-----------------------------------------------------------------------}

type Col     = [Double]
type Weights = ([Col], [Col], [Col])
type Config  = (Col, Weights, Col)


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


iterate_eps :: Double -> Config -> Int -> IO ()
iterate_eps eps (input, ws, targets) n = do
  let (errors, ws') = ann input ws targets
      done          = all (\x -> abs x < eps) errors

  if n `mod` 250 == 0
  then do putStrLn $ "Errors in output layer: " ++ (show errors)
          putStrLn $ "End of iteration "        ++ (show n)
          iterate_eps' done errors eps (input, ws', targets) n
  else iterate_eps' done errors eps (input, ws', targets) n


iterate_eps' :: Bool -> Col -> Double -> Config -> Int -> IO ()
iterate_eps' False err eps config n = iterate_eps eps config (n + 1)
iterate_eps' True  err eps config n = do
  putStrLn   "---------------------------------------------------------"
  putStrLn $ "Final errors in output layer: " ++ (show err)
  putStrLn $ "Total iterations: "             ++ (show n)


main :: IO ()
main = iterate_eps 0.0000000001 config 0
  where
  -- Hardcoded input: 2 input, 3 hidden, 2 output neurons
  input    = [0.50, 0.45]

  -- Weights from 2 input to 3 hidden nodes
  w_input  = [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]

  -- Weight from bias node (connects with hidden layer)
  w_bias = [[0.1], [0.1], [0.1]]

  -- Weights from 3 hidden to 2 output nodes
  w_hidden = [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]

  targets  = [0.05, 0.95]
  -- Of course the numbers are arbitrary

  weights  = (w_input, w_bias, w_hidden)
  config   = (input, weights, targets)
