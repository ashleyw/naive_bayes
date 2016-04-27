defmodule NaiveBayes do
  defstruct vocab: %Vocab{}, data: %Data{}, smoothing: 1, binarized: false, assume_uniform: false

  def new(opts \\ []) do
    binarized = opts[:binarized] || false
    assume_uniform = opts[:assume_uniform] || false
    smoothing = opts[:smoothing] || 1
    {:ok, pid} = Agent.start_link fn ->
      %NaiveBayes{smoothing: smoothing, binarized: binarized, assume_uniform: assume_uniform}
    end
    pid
  end

  def train(pid, tokens, categories) do
    categories = List.flatten [categories]
    Agent.get_and_update pid, fn classifier ->
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
    end
  end

  def classify(pid, tokens) do
    classifier = classifier_instance(pid)
    if classifier.binarized do
      tokens = Enum.uniq(tokens)
    end
    calculate_probabilities(classifier, tokens)
  end

  def purge_less_than(pid, x) do
    Agent.get_and_update pid, fn classifier ->
      {classifier, remove_list} = Enum.reduce(classifier.vocab.tokens, {classifier, []}, fn ({token, _}, {classifier, remove_list}) ->
        case Data.purge_less_than(classifier.data, token, x) do
          false -> []
          data ->
            classifier = put_in(classifier.data, data)
            remove_list = remove_list ++ [token]
        end
        {classifier, remove_list}
      end)

      classifier = Enum.reduce(remove_list, classifier, fn (token, classifier) ->
        vocab = Map.delete(classifier.vocab, token)
        put_in(classifier.vocab, vocab)
      end)

      {:ok, classifier}
    end
  end

  def set_smoothing(pid, x) do
    Agent.get_and_update pid, fn classifier ->
      {:ok, put_in(classifier.smoothing, x)}
    end
  end

  def assume_uniform(pid, bool) do
    Agent.get_and_update pid, fn classifier ->
      {:ok, put_in(classifier.assume_uniform, bool)}
    end
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
