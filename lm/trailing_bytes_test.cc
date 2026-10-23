// A binary model can have other data after it, such as a hash of what it was made for appended
// to the file: it has to load, also when the vocabulary strings are asked for.

#include "model.hh"

#include "enumerate_vocab.hh"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#define BOOST_TEST_MODULE TrailingBytesTest
#include <boost/test/unit_test.hpp>

namespace lm {
namespace ngram {
namespace {

const char *TestLocation() {
  if (boost::unit_test::framework::master_test_suite().argc < 2) {
    return "test.arpa";
  }
  return boost::unit_test::framework::master_test_suite().argv[1];
}

std::string ReadFile(const std::string &name) {
  std::ifstream in(name.c_str(), std::ios::binary);
  BOOST_REQUIRE(in);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void WriteFile(const std::string &name, const std::string &data) {
  std::ofstream out(name.c_str(), std::ios::binary);
  out.write(data.data(), data.size());
  BOOST_REQUIRE(out);
}

class CollectVocab : public EnumerateVocab {
  public:
    void Add(WordIndex index, const StringPiece &str) {
      BOOST_CHECK_EQUAL(words.size(), static_cast<std::size_t>(index));
      words.push_back(str.as_string());
    }
    std::vector<std::string> words;
};

template <class M> double Score(const M &model) {
  const char *sentence[] = {"looking", "on", "a", "little", "the", "more", ",", "loin", "of", "a", "<unk>", "word"};
  double total = 0.0;
  typename M::State state(model.BeginSentenceState()), out;
  for (std::size_t i = 0; i < sizeof(sentence) / sizeof(*sentence); ++i) {
    total += model.FullScore(state, model.GetVocabulary().Index(sentence[i]), out).prob;
    state = out;
  }
  return total + model.FullScore(state, model.GetVocabulary().EndSentence(), out).prob;
}

template <class M> void TrailingBytesTest() {
  const std::string plain = "trailing_test_plain.binary", trailed = "trailing_test_trailed.binary";

  Config build;
  build.messages = NULL;
  build.write_mmap = plain.c_str();
  { M built(TestLocation(), build); }

  CollectVocab expected_vocab;
  Config load;
  load.messages = NULL;
  load.enumerate_vocab = &expected_vocab;
  M reference(plain.c_str(), load);
  const double expected = Score(reference);
  BOOST_CHECK_LT(expected, 0.0);

  const std::string binary = ReadFile(plain);
  const std::string trailers[] = {
    std::string("a"),
    std::string("a trailer holding a hash of what the model was made for"),
    std::string("with\0nulls\0", 11),
    std::string(100000, 'x'),
  };
  const util::LoadMethod methods[] = {util::LAZY, util::POPULATE_OR_READ, util::READ};
  for (std::size_t t = 0; t < sizeof(trailers) / sizeof(*trailers); ++t) {
    WriteFile(trailed, binary + trailers[t]);
    for (std::size_t m = 0; m < sizeof(methods) / sizeof(*methods); ++m) {
      CollectVocab vocab;
      Config config;
      config.messages = NULL;
      config.load_method = methods[m];
      config.enumerate_vocab = &vocab;
      M model(trailed.c_str(), config);
      BOOST_CHECK_EQUAL(expected, Score(model));
      // The vocabulary has to stop at the end of the model, not the end of the file.
      BOOST_CHECK(expected_vocab.words == vocab.words);
      ModelType type;
      BOOST_CHECK(RecognizeBinary(trailed.c_str(), type));
      BOOST_CHECK_EQUAL(M::kModelType, type);
      // And without asking for the vocabulary.
      config.enumerate_vocab = NULL;
      M quiet(trailed.c_str(), config);
      BOOST_CHECK_EQUAL(expected, Score(quiet));
    }
  }

  // A model cut short is still an error.
  WriteFile(trailed, binary.substr(0, binary.size() - 1000));
  {
    CollectVocab vocab;
    Config config;
    config.messages = NULL;
    config.enumerate_vocab = &vocab;
    BOOST_CHECK_THROW(M(trailed.c_str(), config), util::Exception);
  }

  std::remove(plain.c_str());
  std::remove(trailed.c_str());
}

BOOST_AUTO_TEST_CASE(probing) { TrailingBytesTest<ProbingModel>(); }
BOOST_AUTO_TEST_CASE(rest_probing) { TrailingBytesTest<RestProbingModel>(); }
BOOST_AUTO_TEST_CASE(trie) { TrailingBytesTest<TrieModel>(); }
BOOST_AUTO_TEST_CASE(quant_trie) { TrailingBytesTest<QuantTrieModel>(); }
BOOST_AUTO_TEST_CASE(bhiksha_trie) { TrailingBytesTest<ArrayTrieModel>(); }
BOOST_AUTO_TEST_CASE(quant_bhiksha_trie) { TrailingBytesTest<QuantArrayTrieModel>(); }

} // namespace
} // namespace ngram
} // namespace lm
