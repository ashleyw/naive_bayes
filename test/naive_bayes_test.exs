defmodule NaiveBayesTest do
  use ExUnit.Case, async: true

  setup do
      {:ok, nbayes} = NaiveBayes.new
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

    {:ok} = nbayes |> NaiveBayes.purge_less_than(2)

    results = nbayes |> NaiveBayes.classify( ~w(c) )

    assert results["classA"] == 0.5
    assert results["classB"] == 0.5
  end



  test "allows smoothing constant k to be set to any value", context do
    nbayes = context[:nbayes]
    nbayes |> NaiveBayes.train( ~w(a a a c), "classA" )
    nbayes |> NaiveBayes.train( ~w(b b b d), "classB" )

    results = nbayes |> NaiveBayes.classify( ~w(c) )

    prob_k1 = results["classA"]
    {:ok} = nbayes |> NaiveBayes.set_smoothing(5)

    results = nbayes |> NaiveBayes.classify( ~w(c) )
    prob_k5 = results["classA"]
    assert prob_k1 > prob_k5
  end



  test "should allow binarized mode", context do
    nbayes = context[:nbayes]

    train_it = fn ->
      nbayes |> NaiveBayes.train( ~w(a a a a a a a a a a a), "classA" )
      nbayes |> NaiveBayes.train( ~w(b b), "classA" )
      nbayes |> NaiveBayes.train( ~w(a c), "classB" )
      nbayes |> NaiveBayes.train( ~w(a c), "classB" )
      nbayes |> NaiveBayes.train( ~w(a c), "classB" )
    end
    train_it.()
    results = nbayes |> NaiveBayes.classify( ~w(a) )
    assert results["classA"] > 0.5

    {:ok, nbayes} = NaiveBayes.new(binarized: true)

    train_it.()
    results = nbayes |> NaiveBayes.classify( ~w(a) )
    assert results["classB"] > 0.5
  end



  test "should optionally allow class distribution to be assumed uniform", context do
    nbayes = context[:nbayes]

    nbayes |> NaiveBayes.train( ~w(a a a a b), "classA" )
    nbayes |> NaiveBayes.train( ~w(a a a a), "classA" )
    nbayes |> NaiveBayes.train( ~w(a a a a), "classB" )

    results = nbayes |> NaiveBayes.classify( ~w(a) )
    assert results["classA"] > 0.5

    {:ok} = nbayes |> NaiveBayes.assume_uniform(true)

    results = nbayes |> NaiveBayes.classify( ~w(a) )
    assert results["classB"] > 0.5
  end



  test "should pass README example", context do
    nbayes = context[:nbayes]

    stopwords = ~w(a able about above abst accordance according accordingly across act actually added adj affected affecting affects after afterwards again against ah all almost alone along already also although always am among amongst an and announce another any anybody anyhow anymore anyone anything anyway anyways anywhere apparently approximately are aren arent arise around as aside ask asking at auth available away awfully b back be became because become becomes becoming been before beforehand begin beginning beginnings begins behind being believe below beside besides between beyond biol both brief briefly but by c ca came can cannot can't cause causes certain certainly co com come comes contain containing contains could couldnt d date did didn't different do does doesn't doing done don't down downwards due during e each ed edu effect eg eight eighty either else elsewhere end ending enough especially et et-al etc even ever every everybody everyone everything everywhere ex except f far few ff fifth first five fix followed following follows for former formerly forth found four from further furthermore g gave get gets getting give given gives giving go goes gone got gotten h had happens hardly has hasn't have haven't having he hed hence her here hereafter hereby herein heres hereupon hers herself hes hi hid him himself his hither home how howbeit however hundred i id ie if i'll im immediate immediately importance important in inc indeed index information instead into invention inward is isn't it itd it'll its itself i've j just k keep keeps kept kg km know known knows l largely last lately later latter latterly least less lest let lets like liked likely line little 'll look looking looks ltd m made mainly make makes many may maybe me mean means meantime meanwhile merely mg might million miss ml more moreover most mostly mr mrs much mug must my myself n na name namely nay nd near nearly necessarily necessary need needs neither never nevertheless new next nine ninety no nobody non none nonetheless noone nor normally nos not noted nothing now nowhere o obtain obtained obviously of off often oh ok okay old omitted on once one ones only onto or ord other others otherwise ought our ours ourselves out outside over overall owing own p page pages part particular particularly past per perhaps placed please plus poorly possible possibly potentially pp predominantly present previously primarily probably promptly proud provides put q que quickly quite qv r ran rather rd re readily really recent recently ref refs regarding regardless regards related relatively research respectively resulted resulting results right run s said same saw say saying says sec section see seeing seem seemed seeming seems seen self selves sent seven several shall she shed she'll shes should shouldn't show showed shown showns shows significant significantly similar similarly since six slightly so some somebody somehow someone somethan something sometime sometimes somewhat somewhere soon sorry specifically specified specify specifying still stop strongly sub substantially successfully such sufficiently suggest sup sure than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves)
    tokenize = fn s ->
      (s |> String.downcase |> String.replace(~r/[^a-z ]/, "") |> String.split(~r/\s+/)) -- stopwords
    end

    nbayes |> NaiveBayes.train( tokenize.("You need to buy some Viagra"), "SPAM" )
    nbayes |> NaiveBayes.train( tokenize.("This is not spam, just a letter to Bob."), "HAM" )
    nbayes |> NaiveBayes.train( tokenize.("Hey Oasic, Do you offer consulting?"), "HAM" )
    nbayes |> NaiveBayes.train( tokenize.("You should buy this stock"), "SPAM" )

    results = nbayes |> NaiveBayes.classify( tokenize.("Now is the time to buy Viagra cheaply and discreetly") )
    assert results["SPAM"] > 0.5
  end



  test "train/3 should return {:ok}", context do
    nbayes = context[:nbayes]
    {status} = nbayes |> NaiveBayes.train( ~w(a a a a b), "classA" )
    assert status == :ok
  end



  test "train/3 should require n>0 tokens", context do
    nbayes = context[:nbayes]
    {status} = nbayes |> NaiveBayes.train( [], "classA" )
    assert status == :error
  end


  test "train/3 should require n>0 categories", context do
    nbayes = context[:nbayes]
    {status} = nbayes |> NaiveBayes.train( ~w(a a a a b), [] )
    assert status == :error
  end

end
