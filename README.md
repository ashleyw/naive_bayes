# NaiveBayes [![Build Status](https://travis-ci.org/ashleyw/naive_bayes.svg?branch=master)](https://travis-ci.org/ashleyw/naive_bayes)

## Features

- works with all types of tokens, not just text
- allows purging of low-frequency tokens (for performance)
- uses log probabilities

## Usage

```elixir
nbayes = NaiveBayes.new
nbayes |> NaiveBayes.train( ~w(a b c d e f g), "classA" )
nbayes |> NaiveBayes.train( ~w(a b c d e f g), "classB" )

results = nbayes |> NaiveBayes.classify( ~w(a b c) )

IO.inspect results
# => %{"classA" => 0.5, "classB" => 0.5}
```

See `test/naive_bayes_test.ex` for more examples.

## Installation

```elixir
def deps do
  [{:naive_bayes, "~> 0.0.1"}]
end
```
