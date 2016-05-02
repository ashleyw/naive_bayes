defmodule NaiveBayes do
  @moduledoc """
  An implementation of Naive Bayes
  """
  defstruct vocab: %Vocab{}, data: %Data{}, smoothing: 1, binarized: false, assume_uniform: false

  @doc """
  Initializes a new NaiveBayes agent

  Returns `{:ok, pid}`.

  ## Examples

      iex> {:ok, nbayes} = NaiveBayes.new(binarized: false, assume_uniform: true, smoothing: 2)
      {:ok, #PID<0.137.0>}

  """
  def new(opts \\ []) do
    binarized = opts[:binarized] || false
    assume_uniform = opts[:assume_uniform] || false
    smoothing = opts[:smoothing] || 1
    {:ok, pid} = Agent.start_link fn ->
      %NaiveBayes{smoothing: smoothing, binarized: binarized, assume_uniform: assume_uniform}
    end
    {:ok, pid}
  end


  @doc """
  Trains the naive bayes instance given a list of tokens and categories

  Returns `{:ok}` or `{:error}`

  ## Examples

      iex> {:ok, nbayes} = NaiveBayes.new
      {:ok, #PID<0.137.0>}
      iex> nbayes |> NaiveBayes.train( ["a", "b", "c"], "classA" )
      {:ok}

  """
  def train(pid, tokens, categories) do
    categories = List.flatten [categories]

    case Enum.count(tokens) > 0 && Enum.count(categories) > 0 do
      true ->
        Agent.get_and_update(pid, fn classifier ->
          if classifier.binarized do
            tokens = Enum.uniq(tokens)
          end
          classifier = Enum.reduce(categories, classifier, fn(category, classifier) ->
            classifier = put_in(classifier.data, Data.increment_examples(classifier.data, category))
            Enum.reduce(tokens, classifier, fn(token, classifier) ->
              classifier = put_in(classifier.data, Data.add_token_to_category(classifier.data, category, token))
              put_in(classifier.vocab, Vocab.seen_token(classifier.vocab, token))
            end)
          end)
          {:ok, classifier}
        end)
        :ok
      false ->
        :error
    end
  end


  @doc """
  Returns a list of probabilities of classes given a list of tokens.

  ## Examples

      iex> results = nbayes |> NaiveBayes.classify( ["a", "b", "c"] )
      %{"HAM" => 0.4832633319857435, "SPAM" => 0.5167366680142564}

  """
  def classify(pid, tokens) do
    classifier = classifier_instance(pid)
    if classifier.binarized do
      tokens = Enum.uniq(tokens)
    end
    calculate_probabilities(classifier, tokens)
  end


  @doc """
  Allows removal of low frequency words that increase processing time and may overfit

  Returns `{:ok}`

  ## Examples

      iex> nbayes |> NaiveBayes.purge_less_than(5)
      :ok

  """
  def purge_less_than(pid, x) do
    Agent.get_and_update(pid, fn classifier ->
      {classifier, remove_list} = Enum.reduce(classifier.vocab.tokens, {classifier, []}, fn ({token, _}, {classifier, remove_list}) ->
        case Data.purge_less_than(classifier.data, token, x) do
          false -> nil
          data ->
            classifier = put_in(classifier.data, data)
            remove_list = remove_list ++ [token]
        end
        {classifier, remove_list}
      end)

      classifier = Enum.reduce(remove_list, classifier, fn (token, classifier) ->
        put_in(classifier.vocab, Vocab.remove_token(classifier.vocab, token))
      end)

      {:ok, classifier}
    end, 3600*24*30*1000) # don't timeout
    :ok
  end


  @doc """
  Increase smoothing constant to dampen the effect of the rare tokens

  Returns `{:ok}`

  ## Examples

      iex> nbayes |> NaiveBayes.set_smoothing(2)
      :ok

  """
  def set_smoothing(pid, x) do
    Agent.get_and_update pid, fn classifier ->
      {:ok, put_in(classifier.smoothing, x)}
    end
    :ok
  end


  @doc """
  Set the assume_uniform constant.

  Returns `{:ok}`

  ## Examples

      iex> nbayes |> NaiveBayes.assume_uniform(true)
      :ok

  """
  def assume_uniform(pid, bool) do
    Agent.get_and_update pid, fn classifier ->
      {:ok, put_in(classifier.assume_uniform, bool)}
    end
    :ok
  end

  defp calculate_probabilities(classifier, tokens) do
    v_size = Enum.count(classifier.vocab.tokens)
    total_example_count = Data.total_examples(classifier.data)

    prob_numerator = Enum.reduce(classifier.data.categories, %{}, fn ({cat_name, cat_data}, probs) ->
      cat_prob = case classifier.assume_uniform do
        true -> :math.log(1 / Enum.count(classifier.data.categories))
        false -> :math.log(Data.example_count(cat_data) / total_example_count)
      end

      denominator = (cat_data[:total_tokens] + classifier.smoothing * v_size)
      log_probs = Enum.reduce(tokens, 0, fn (token, log_probs) ->
        numerator = (cat_data[:tokens][token] || 0) + classifier.smoothing
        log_probs + :math.log( numerator / denominator )
      end)
      put_in(probs[cat_name], log_probs + cat_prob)
    end)
    normalize(prob_numerator)
  end

  defp normalize(prob_numerator) do
    normalizer = Enum.reduce(prob_numerator, 0, fn ({_, numerator}, normalizer) ->
      normalizer + numerator
    end)

    {intermed, renormalizer} = Enum.reduce(prob_numerator, {%{}, 0}, fn ({cat, numerator}, {intermed, renormalizer}) ->
      r = normalizer / numerator
      intermed = put_in(intermed, [cat], r)
      renormalizer = renormalizer + r
      {intermed, renormalizer}
    end)

    Enum.reduce(intermed, %{}, fn ({cat, value}, final_probs) ->
      put_in(final_probs, [cat], value / renormalizer)
    end)
  end

  defp classifier_instance(pid) do
    Agent.get pid, fn c -> c end
  end
end
