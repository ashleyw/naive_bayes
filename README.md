# NaiveBayes [![Build Status](https://travis-ci.org/ashleyw/naive_bayes.svg?branch=master)](https://travis-ci.org/ashleyw/naive_bayes) [![Hex.pm](https://img.shields.io/hexpm/v/naive_bayes.svg?maxAge=0)](https://hex.pm/packages/naive_bayes)

## Features

- works with all types of tokens, not just text
- allows purging of low-frequency tokens (for performance)
- uses log probabilities to avoid underflow
- allows prior distribution on classes to be assumed uniform
- customizable constant value for Laplacian smoothing
- allows for multiple categories
- optional binarized mode

## Usage

```elixir
{:ok, nbayes} = NaiveBayes.new

tokenize = fn s ->
  s |> String.downcase |> String.replace(~r/[^a-z ]/, "") |> String.split(~r/\s+/)
end

nbayes |> NaiveBayes.train( tokenize.("You need to buy some Viagra"), "SPAM" )
nbayes |> NaiveBayes.train( tokenize.("This is not spam, just a letter to Bob."), "HAM" )
nbayes |> NaiveBayes.train( tokenize.("Hey Oasic, Do you offer consulting?"), "HAM" )
nbayes |> NaiveBayes.train( tokenize.("You should buy this stock"), "SPAM" )

results = nbayes |> NaiveBayes.classify( tokenize.("Now is the time to buy Viagra cheaply and discreetly") )

IO.inspect results
# => %{"HAM" => 0.4832633319857435, "SPAM" => 0.5167366680142564}
```

See the [docs](https://hexdocs.pm/naive_bayes/NaiveBayes.html) or [`test/naive_bayes_test.ex`](https://github.com/ashleyw/naive_bayes/blob/master/test/naive_bayes_test.exs) for more examples.

## Installation

```elixir
def deps do
  [{:naive_bayes, "~> 0.1.3"}]
end
```
