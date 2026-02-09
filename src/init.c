#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

extern SEXP cpp_t3_decode(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"cpp_t3_decode", (DL_FUNC) &cpp_t3_decode, 11},
    {NULL, NULL, 0}
};

void R_init_chatterbox(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
