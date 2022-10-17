from tree_sitter import Language, Parser

JAVA_LANGUAGE = Language("./my-languages.so", "c_sharp")
parser = Parser()
parser.set_language(JAVA_LANGUAGE)
