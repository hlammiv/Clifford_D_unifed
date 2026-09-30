# Draft corrections: ESA_CliffordD_fixed.tex

*2026-09-30. Target file: `ESA_CliffordD_fixed.tex` inside `Clifford_D_Householder/Synthesis_of_Single_Qutrit_Circuits_from_Clifford_D.zip` (745 lines, dated 2026-05-05). The compiled `summary/Synthesis_of_Single_Qutrit_Circuits_from_Clifford_D.pdf` (May 29) has the same text, so equation numbers below refer to that PDF.*

- Line numbers are those of the .tex. Neither the zip nor the PDF was modified.
- Each patch gives the location, the exact **Find** string (copy-paste safe, unique in the file) and the **Replace** string.
- The patches assume the convention recommended in `convention_memo.md`: the signed 𝒟 of Kalra/Evra–Parzanchevski, R charged, and per-type counts plus T-cost.
- Macros already in the preamble are used (`\bR`, `\bD`, `\CD`, `\Zn`, `\Diag`, `\sde`). Patch 0 adds one macro, `\cD`.

**Severity:**
- **E** = mathematically wrong.
- **I** = inconsistent with the rest of the paper or the code.
- **C** = clarity or typo.

---

## Patch 0 (preamble, after l.165), I
Add a macro for the signed diagonal set used throughout.

Find:
```latex
\newcommand{\Znt}{\mathbb{Z}[\zeta_9,\frac{1}{3}]}
```
Replace:
```latex
\newcommand{\Znt}{\mathbb{Z}[\zeta_9,\frac{1}{3}]}
\newcommand{\cD}{\mathcal{D}}
```

## Patch 1 (l.232, Introduction: definition of C+D), I/E
**Problem.** The Intro defines C+D with *unsigned* ninth-root diagonals. Three later parts of the paper need the signed set 𝒟 of Kalra et al. and Evra–Parzanchevski, which contains R:
- the identification C+D = U(3, Z[ζ₉,1/3]) at l.241;
- the syllable (32), which contains R^ε;
- l.597, l.695 and l.707, which all use ±ζ₉^k.

Our reducer also fails on 10/10 test matrices when R is removed (`unified/nick_test/no_r_test.py`).

Find:
```latex
Here, $\CD$ is the qutrit Clifford group extended by the diagonal matrices whose diagonal entries are ninth roots of unity.
```
Replace:
```latex
Here, following~\cite{Kal,EvPar}, $\CD$ is the qutrit Clifford group extended by the set
$\cD=\{\Diag(\pm\zeta_9^{a},\pm\zeta_9^{b},\pm\zeta_9^{c})\}$ of diagonal matrices whose entries are \emph{signed} ninth roots of unity (equivalently, $18$th roots of unity). Note that $\cD$ contains the metaplectic gate $\bR=\Diag(1,1,-1)$, so $\CR\subseteq\CD$.
```

## Patch 2 (l.241, the EP citation), I
**Problem.** The statement is true only for the signed set. Add a clause saying so.

Find:
```latex
This fact was established in \cite[Theorem 2.8]{EvPar},
```
Replace:
```latex
This fact was established in \cite[Theorem 2.8]{EvPar} for the signed diagonal set $\cD$ defined above; it would fail if $\cD$ were restricted to unsigned ninth roots of unity, since $\bR\in\U(3,\Z[\zeta_9,\frac13])$ but every such unsigned word has determinant in $\mu_9$,
```
**Note.** The determinant argument excludes R and ζ^k R exactly, but not −ζ^k R (see `unified/cd_approx_cr_notes.md` §9). If the authors want an airtight statement, use the weaker wording "…it relies on the signed set $\cD$ defined above, which contains $\bR$," and drop the determinant clause.

## Patch 3 (l.329, the Householder target), C (typo with an R consequence)
**Problem.** The target is written with −1 in the (3,3) slot. The abstract (l.215), the Intro (l.228) and the identity R_u = I − 2uu* all require +1, since (I − 2uu*)₃₃ = 1. The −1 version differs from the target by exactly R.

Find:
```latex
Recall that we are trying to find an approximation of the diagonal matrix $R_{(0,1)}^Z(\theta) = \mbox{Diag}(e^{-i\theta/2}, e^{i\theta/2},-1)$
```
Replace:
```latex
Recall that we are trying to find an approximation of the diagonal matrix $R_{(0,1)}^Z(\theta) = \mbox{Diag}(e^{-i\theta/2}, e^{i\theta/2},1)$
```

## Patch 4 (l.584, the C+R comparison: "R introduces one power of χ"), E
**Problems.**
- R has unit entries (±1), so applying it changes no sde. The denominators come from H.
- The bound "sde_χ = k ⇒ at most k applications of R" fails at k = 0: R itself has sde_χ = 0 and needs one R.
- More generally the final monomial can carry one R.

Find:
```latex
Crucially, the non-Clifford gate $\bR = \Diag(1,1,-1)$ introduces exactly one power of $\chi$ into the denominator of a matrix entry when applied. Consequently, $\sde_\chi$ directly bounds the $\bR$-gate count: an element of $\U(3,\Z[\omega,\chi^{-1}])$ with $\sde_\chi = k$ requires at most $k$ applications of $\bR$.
```
Replace:
```latex
Crucially, exact synthesis in $\CR$ proceeds by syllables $H\,D\,\bR^{\epsilon}X^{\delta}$, each containing at most one $\bR$ and lowering $\sde_\chi$ by exactly one (Property~P of~\cite{Kal}); the Hadamard supplies the power of $\chi$, while $\bR$ (whose entries are units) prevents consecutive Cliffords from collapsing. Consequently, $\sde_\chi$ directly bounds the $\bR$-gate count: an element of $\U(3,\Z[\omega,\chi^{-1}])$ with $\sde_\chi = k$ requires at most $k+1$ applications of $\bR$, the extra one accounting for a possible sign in the final monomial (e.g.\ $\bR$ itself has $\sde_\chi=0$).
```

## Patch 5 (l.586, "sde_χ = 0 exactly characterizes the Clifford group"), E
**Problem.** R has sde_χ = 0 and is not Clifford. The Clifford diagonals D(a,b,c) give only powers of ω, not the signs ±1.

Find:
```latex
The only units in $\Z[\omega]$ are the six third roots of unity $\{\pm 1, \pm \omega, \pm \omega^2\}$, which are precisely the phases available to the qutrit Clifford diagonal gates $D(a,b,c) = \Diag(\omega^a, \omega^b, \omega^c)$. Thus $\sde_\chi = 0$ exactly characterizes the qutrit Clifford group in the $\CR$ setting.
```
Replace:
```latex
The only units in $\Z[\omega]$ are the six sixth roots of unity $\{\pm 1, \pm \omega, \pm \omega^2\}$. The qutrit Clifford diagonal gates $D(a,b,c) = \Diag(\omega^a, \omega^b, \omega^c)$ supply the powers of $\omega$, but not relative signs. Thus $\sde_\chi = 0$ characterizes the monomial group generated by the Clifford monomials and $\bR$: such a matrix is Clifford (up to a global sign) if its nonzero entries carry a common sign, and is a Clifford times a single $\bR$ otherwise.
```

## Patch 6 (l.590, "the non-Clifford gate D"), I
**Problems.**
- The text calls diag(ζ,1,ζ⁻¹), the qutrit T, "the" non-Clifford gate. The paper counts over the whole set 𝒟.
- There is no statement anywhere about where 𝒟's elements sit in the Clifford hierarchy.

The draft never makes a Clifford-hierarchy claim. Kalra et al. 2311.08696 say "a D gate is in the Clifford hierarchy". That is true only for unsigned D and must not be imported. The replacement adds the correct classification.

Find:
```latex
The non-Clifford gate $\bD = \Diag(\zeta_9, 1, \zeta_9^{-1})$ has entries that are \emph{units} in $\Zn$: both $\zeta_9$ and $\zeta_9^{-1} = \zeta_9^8$ have $\sde_\chi = 0$. Therefore, applying $\bD$ to a unitary matrix \emph{does not change $\sde_\chi$ of any entry}.
```
Replace:
```latex
Every element of $\cD$ has entries that are \emph{units} in $\Zn$ (signed ninth roots of unity have $\sde_\chi = 0$). Therefore, applying any $D\in\cD$ to a unitary matrix \emph{does not change $\sde_\chi$ of any entry}. Physically, the non-Clifford elements of $\cD$ fall into three classes, up to Clifford gates and a global phase: (i) \emph{T-type} diagonals $\Diag(\zeta_9^a,\zeta_9^b,\zeta_9^c)$ with $a+b+c\equiv 0\pmod 3$, each equal to $\bD^{\pm1}$ with $\bD=\Diag(\zeta_9,1,\zeta_9^{-1})$ (the qutrit $T$ gate), at the third level of the Clifford hierarchy; (ii) \emph{level-4} diagonals with $a+b+c\not\equiv 0\pmod 3$, each a single-position phase $\Diag(1,1,\zeta_9^{\pm1})$ up to Clifford; and (iii) diagonals with mixed signs, which equal an unsigned diagonal times $\bR$. Since diagonal gates in the qutrit Clifford hierarchy have phases that are $3^m$-th roots of unity~\cite{CuiGottesmanKrishna}, $\bR$ and every mixed-sign element of $\cD$ lie in \emph{no} level of the hierarchy.
```
Add a bib entry: `\bibitem{CuiGottesmanKrishna} S.~X.~Cui, D.~Gottesman, A.~Krishna, \textit{Diagonal gates in the Clifford hierarchy}, Phys.\ Rev.\ A \textbf{95}, 012329 (2017), arXiv:1608.06596.`

Before submission, verify the level-4 statement in (ii) against a reference or include our own check. At present it rests on an internal numerical check (`cd_ft_cost_results_2026-09-30.md` §2).

## Patch 7 (l.603, Remark), C/I
**Problems.**
- The Clifford condition should be taken up to a global phase.
- Signs should be mentioned.

Find:
```latex
but is in the Clifford group only when $a \equiv b \equiv c \equiv 0 \pmod{3}$. Thus, $\sde_\chi = 0$ includes unitaries requiring $\bD$ gates.
```
Replace:
```latex
but is Clifford (up to the global phase $\zeta_9^{a}$) only when $a \equiv b \equiv c \pmod{3}$; a signed diagonal $\Diag(\pm\zeta_9^a,\pm\zeta_9^b,\pm\zeta_9^c)$ with mixed signs additionally contains a factor $\bR$. Thus, $\sde_\chi = 0$ includes unitaries requiring non-Clifford elements of $\cD$, including $\bR$.
```

## Patch 8 (l.606, the wrong "is Clifford" example), E (required)
**Problem.** −ζ₉⁶ = −ω² and −ζ₉³ = −ω. So

Diag(−ζ₉⁶, −ζ₉³, 1) = Diag(−ω², −ω, 1) = −Diag(ω², ω, −1) = −Diag(ω², ω, 1)·R,

which is a Clifford times R up to a global sign, and not Clifford. It lies in no level of the Clifford hierarchy. The first example is also only the D gate up to a cyclic relabelling: X·Diag(ζ,1,ζ⁻¹)·X† = Diag(ζ⁻¹, ζ, 1).

Find:
```latex
As a concrete example, the unitary $\Diag(\zeta_9^{-1}, \zeta_9, 1)$ has $\sde_\chi = 0$ but is \emph{not} a Clifford gate---it is the $\bD$ gate itself. Conversely, $\Diag(-\zeta_9^6, -\zeta_9^3, 1) = \Diag(\omega^2, \omega, 1)$ also has $\sde_\chi = 0$ and \emph{is} Clifford, since $\zeta_9^3 = \omega$ and $\zeta_9^6 = \omega^2$.
```
Replace:
```latex
As a concrete example, the unitary $\Diag(\zeta_9^{-1}, \zeta_9, 1)$ has $\sde_\chi = 0$ but is \emph{not} a Clifford gate---it is the $\bD$ gate up to a cyclic relabelling of the basis. Conversely, $\Diag(\zeta_9^6, \zeta_9^3, 1) = \Diag(\omega^2, \omega, 1)$ also has $\sde_\chi = 0$ and \emph{is} Clifford, since $\zeta_9^3 = \omega$ and $\zeta_9^6 = \omega^2$. Signs matter, however: $\Diag(-\zeta_9^6, -\zeta_9^3, 1) = -\Diag(\omega^2,\omega,-1) = -\Diag(\omega^2,\omega,1)\,\bR$ also has $\sde_\chi=0$ but is a Clifford gate times $\bR$, and is not Clifford.
```

## Patch 9 (l.610, consequences for the search), I
Find:
```latex
A solution found at $\sde_3 = 0$ may require $\bD$ gates (if its diagonal phases are 9th roots of unity that are not cube roots of unity),
```
Replace:
```latex
A solution found at $\sde_3 = 0$ may require non-Clifford gates (T-type or level-4 diagonals if its phases are 9th roots of unity that are not cube roots of unity, and an $\bR$ if its entries carry mixed signs),
```

## Patch 10 (l.626, after the syllable (32)), I
**Problem.** The syllable uses R, but the draft never says it is a member of the gate set or that it is charged.

Find:
```latex
$\bR = \Diag(1,1,-1)$, and $X$ is the cyclic shift, such that $\sde_\chi(gV) < \sde_\chi(V)$.
```
Replace:
```latex
$\bR = \Diag(1,1,-1)\in\cD$, and $X$ is the cyclic shift, such that $\sde_\chi(gV) < \sde_\chi(V)$. The factor $\bR^{\epsilon}$ is Kalra's sign normalization of the leading residues and is not optional: restricting the syllables to $\epsilon=0$ makes the reduction stall on every matrix we tested. Each syllable with $\epsilon=1$ is therefore charged one $\bR$.
```

## Patch 11 (Algorithm 6, l.671–699: counting), E/I
**Problems.**
- Alg. 6 charges nothing for R^ε.
- It charges nothing for a mixed-sign residual monomial.
- It counts residual phases per entry without first removing a global phase ζ₉^k. For example, Diag(1, ζ, ζ) would be counted as 2, but it is ζ·Diag(ζ⁻¹, 1, 1), a single level-4 phase.
- It returns one number where three are needed.

Find (l.671):
```latex
\State \textbf{Returns:} Sequence of syllables $S$ and $\bD$-gate count $N_{\bD}$
```
Replace:
```latex
\State \textbf{Returns:} Sequence of syllables $S$ and gate counts $(n_T, n_4, n_R)$
```
Find (l.674):
```latex
\State $S \gets$ empty list; $N_{\bD} \gets 0$
```
Replace:
```latex
\State $S \gets$ empty list; $(n_T,n_4,n_R) \gets (0,0,0)$
```
Find (l.680):
```latex
\State Update $N_{\bD}$ from $D$; \textbf{go to} While
```
Replace:
```latex
\State Classify $D$ (Clifford / T-type / level-4) and update $n_T,n_4$; $n_R \gets n_R+\epsilon$; \textbf{go to} While
```
Find (l.688):
```latex
\State Update $s, N_{\bD}$; \textbf{go to} While
```
Replace:
```latex
\State Update $s$ and $(n_T,n_4,n_R)$ from $g_1,g_2$; \textbf{go to} While
```
Find (l.694–699):
```latex
\State \textbf{// Count residual D gates at $\sde_\chi = 0$}
\For{each nonzero entry $V_{ij} = \pm\zeta_9^k$}
\If{$k \not\equiv 0 \pmod{3}$}
\State $N_{\bD} \gets N_{\bD} + 1$
\EndIf
\EndFor
\State \Return $(S, N_{\bD})$
```
Replace:
```latex
\State \textbf{// Residual monomial at $\sde_\chi = 0$: entries $s_i\zeta_9^{k_i}$, $s_i=\pm1$}
\State Remove a global phase so that the signs $s_i$ are as uniform as possible
\If{the $s_i$ are not all equal} \State $n_R \gets n_R + 1$ \EndIf
\State Classify $\Diag(\zeta_9^{k_0},\zeta_9^{k_1},\zeta_9^{k_2})$ modulo Clifford diagonals and a global phase $\zeta_9^{m}$; update $n_T$ or $n_4$
\State \Return $(S, n_T, n_4, n_R)$
```
Also add after the algorithm:
```latex
We report $(n_T,n_4,n_R)$ separately. For comparison with the $\bR$-count of~\cite{Gust} we also quote the phase count $N_\varphi = 2n_T+n_4+n_R$, which charges one unit per ninth-root phase with $3\nmid k$ and one per $\bR$; note that it charges a $T$ gate twice. On hardware whose only non-Clifford resource is the $T$ state we quote the $T$-cost $n_T+7n_4+7n_R$ (Sec.~X).
```

## Patch 12 (l.707, residual monomial), I
Find:
```latex
so an entry $\pm\zeta_9^k$ with $k \not\equiv 0 \pmod{3}$ requires at least one $\bD$ gate.
```
Replace:
```latex
so, after removing a global phase, an entry $\pm\zeta_9^k$ with $k \not\equiv 0 \pmod{3}$ requires a non-Clifford diagonal. Moreover, a Clifford monomial has entries of uniform sign (up to a global $-1$), so a residual monomial whose entries carry mixed signs requires exactly one $\bR$.
```

## Patch 13 (eq. (37), l.709–712: the sde₃ = 0 Householder case), E
**Problem.** With α = −ζ₉^a, V = Diag(−ζ₉^{−a}, −ζ₉^{a}, 1) = −Diag(ζ₉^{−a}, ζ₉^{a}, 1)·R, which needs one R. Eq. (37) charges 0 R for every α. For α = −1 in particular, it declares Diag(−1,−1,1) = −R free.

Diag(ζ₉^{−a}, ζ₉^{a}, 1) has exponent sum 0, so it is T-type. It is one T gate, although the draft's "2" charges it two phases.

Find:
```latex
For the special case of the Householder search at $\sde_3 = 0$ (where no peeling occurs), the unitary is always diagonal: $V = \Diag(\bar\alpha, \alpha, 1)$ for some unit $\alpha = \pm\zeta_9^a \in \Zn$. The $\bD$-count is then
\begin{equation}
N_{\bD} = \begin{cases} 0 & \text{if } a \equiv 0 \pmod{3},\\ 2 & \text{otherwise.}\end{cases}
\end{equation}
```
Replace:
```latex
For the special case of the Householder search at $\sde_3 = 0$ (where no peeling occurs), the unitary is always diagonal: $V = \Diag(\bar\alpha, \alpha, 1)$ for some unit $\alpha = s\,\zeta_9^a \in \Zn$ with $s=\pm1$. Writing $V = s\,\Diag(\zeta_9^{-a},\zeta_9^{a},s)$, the non-Clifford content is
\begin{equation}
n_T = \begin{cases} 0 & a \equiv 0 \pmod{3},\\ 1 & \text{otherwise,}\end{cases}
\qquad
n_R = \begin{cases} 0 & s=+1,\\ 1 & s=-1,\end{cases}
\end{equation}
since $\Diag(\zeta_9^{-a},\zeta_9^{a},1)$ is T-type ($\bD^{\pm1}$ up to Clifford) and a relative sign costs one $\bR$. In phase units, $N_\varphi = 2n_T+n_R$.
```

## Patch 14 (Abstract, l.215), C, optional
Add one sentence so the convention is visible up front.

Find:
```latex
We present a post-processing decomposition algorithm,
```
Replace:
```latex
Throughout, $\CD$ uses the signed diagonal set of Kalra et al.\ and Evra--Parzanchevski, which contains the metaplectic gate $\bR=\Diag(1,1,-1)$; we find that $\bR$ is required in exact decompositions and we charge it. We present a post-processing decomposition algorithm,
```

---

## Not patched: checks for the authors

1. **l.640:** the C+R syllable count "3³ × 2² = 108" does not obviously match H·D·R^ε·X^δ with δ ∈ {0,1,2}, which would give 27·2·3 = 162. Check it against Gustafson.
2. **l.349:** "R_v is a matrix with sde₃ at most f²" looks odd; one expects at most 2f or f. Check it. This is unrelated to the R convention.
3. **l.638:** "the monomial group generated by Pauli gates and 𝒟 gates" is correct once 𝒟 is defined as signed (Patch 1). No change is needed if Patch 1 is applied.
4. **Prop. VIII.1 (l.593):** holds for all of signed 𝒟, including R. No change is needed after Patch 1.
5. **Any results section or figure** that quotes "N_D" from runs made before 2026-09-30 undercounts R by up to 1 per rotation (the residual-R bug, `cd_ft_cost_results_2026-09-30.md` §5). Regenerate these before inserting them.
