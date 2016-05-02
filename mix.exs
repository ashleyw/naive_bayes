defmodule NaiveBayes.Mixfile do
  use Mix.Project

  def project do
    [
      app: :naive_bayes,
      version: "0.1.3",
      elixir: "~> 1.2",
      package: package,
      description: description,
      build_embedded: Mix.env == :prod,
      start_permanent: Mix.env == :prod,
      deps: deps
   ]
  end

  def application do
    [applications: []]
  end

  defp deps do
    [{:earmark, "~> 0.1", only: :dev},
     {:ex_doc, "~> 0.11", only: :dev}]
  end

  defp description do
   """
   An Elixir implementation of Naive Bayes
   """
 end

  defp package do
    [
      maintainers: ["Ashley Williams"],
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/ashleyw/naive_bayes"}
    ]
  end
end
