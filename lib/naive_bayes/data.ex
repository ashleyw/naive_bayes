defmodule Data do
  defstruct categories: %{}

  def increment_examples(data, category) do
    if get_in(data.categories, [category]) == nil do
      data = put_in(data.categories[category], %{})
    end
    data = update_in(data.categories[category][:examples], fn e ->
      case e do
        nil -> 1
        _ -> e + 1
      end
    end)
    data
  end

  def add_token_to_category(data, category, token) do
    if get_in(data.categories, [category]) == nil do
      data = put_in(data.categories[category], %{})
    end
    if get_in(data.categories, [category, :tokens]) == nil do
      data = put_in(data.categories[category][:tokens], %{})
    end
    data = update_in(data.categories[category][:tokens][token], fn e ->
      case e do
        nil -> 1
        _ -> e + 1
      end
    end)
    data = update_in(data.categories[category][:total_tokens], fn e ->
      case e do
        nil -> 1
        _ -> e + 1
      end
    end)
    data
  end

  def total_examples(data) do
    Enum.reduce(data.categories, 0, fn ({_, category}, sum) ->
      sum + example_count(category)
    end)
  end

  def example_count(category) do
    category[:examples]
  end

  def purge_less_than(data, token, x) do
    case token_count_across_categories(data, token) >= x do
      true -> false
      false ->
        Enum.reduce(data.categories, data, fn (category, data) ->
          delete_token_from_category(data, category, token)
        end)
    end
  end

  def token_count_across_categories(data, token) do
    Enum.reduce(data.categories, 0, fn ({_, cat_data}, sum) ->
      sum + (cat_data[:tokens][token] || 0)
    end)
  end

  def delete_token_from_category(data, {cat_name, cat_data} = _, token) do
    count = cat_data[:tokens][token] || 0
    tokens = Map.delete(data.categories[cat_name][:tokens], token)
    data = put_in(data.categories[cat_name][:tokens], tokens)
    data = update_in(data.categories[cat_name][:total_tokens], fn e ->
      e - count
    end)
    data
  end
end
