defmodule NaiveBayesTest do
  use ExUnit.Case
  doctest NaiveBayes

  setup do
      nbayes = NaiveBayes.new
      {:ok, nbayes: nbayes}
    end


  test "should assign equal probability to each class", context do
    nbayes = context[:nbayes]

    nbayes |> NaiveBayes.train( ~w(a b c d e f g), "classA" )
    nbayes |> NaiveBayes.train( ~w(a b c d e f g), "classB" )
    results = nbayes |> NaiveBayes.classify( ~w(a b c) )

    assert results["classA"] == 0.5
    assert results["classB"] == 0.5
  end



  test "should allow multiple categories", context do
    nbayes = context[:nbayes]

    nbayes |> NaiveBayes.train( ~w(a b c), ~w(classA classB classC classD classE) )
    nbayes |> NaiveBayes.train( ~w(b c e), ~w(classA classB classC classD classE) )
    nbayes |> NaiveBayes.train( ~w(x y z), ~w(classF) )
    results = nbayes |> NaiveBayes.classify( ~w(w x y) )

    assert results["classF"] > results["classD"]
  end



  test "should handle more than 2 classes", context do
    nbayes = context[:nbayes]

    nbayes |> NaiveBayes.train( ~w(a a a a), "classA" )
    nbayes |> NaiveBayes.train( ~w(b b b b), "classB" )
    nbayes |> NaiveBayes.train( ~w(c c c), "classC" )
    results = nbayes |> NaiveBayes.classify( ~w(a a a a b c) )

    assert results["classA"] >= 0.4
    assert results["classB"] <= 0.3
    assert results["classC"] <= 0.3
  end



  test "should use smoothing by default to eliminate errors w/division by zero", context do
    nbayes = context[:nbayes]

    nbayes |> NaiveBayes.train( ~w(a a a a), "classA" )
    nbayes |> NaiveBayes.train( ~w(b b b b), "classB" )
    results = nbayes |> NaiveBayes.classify( ~w(x y z) )

    assert results[:classA] >= 0.0
    assert results[:classB] >= 0.0
  end



  test "works on all tokens - not just strings", context do
    nbayes = context[:nbayes]

    nbayes |> NaiveBayes.train( [1, 2, 3], "low" )
    nbayes |> NaiveBayes.train( [5, 6, 7], "high" )
    results = nbayes |> NaiveBayes.classify( [2] )

    assert results["low"] > results["high"]
  end



  test "should optionally purge low frequency data", context do
    nbayes = context[:nbayes]

    (1..100) |> Enum.to_list |> Enum.each(fn _ ->
      nbayes |> NaiveBayes.train( ~w(a a a a), "classA" )
      nbayes |> NaiveBayes.train( ~w(b b b b), "classB" )
    end)

    nbayes |> NaiveBayes.train( ~w(a), "classA" )
    nbayes |> NaiveBayes.train( ~w(c b), "classB" )

    results = nbayes |> NaiveBayes.classify( ~w(c) )

    assert results["classB"] > 0.5

    nbayes |> NaiveBayes.purge_less_than(2)

    results = nbayes |> NaiveBayes.classify( ~w(c) )

    assert results["classA"] == 0.5
    assert results["classB"] == 0.5
  end
end
