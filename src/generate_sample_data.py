"""
Generate synthetic legal case documents for testing the RAG pipeline.
Run: python generate_sample_data.py
This creates sample .txt files (treated as plain documents) in data/
"""

import json
from pathlib import Path

SAMPLE_CASES = [
    {
        "filename": "force_majeure_cases.txt",
        "content": """LEGAL MEMORANDUM — FORCE MAJEURE IN SUPPLY CONTRACTS
Source: Commercial Law Review, Vol. 12 | Page: 1

Force majeure clauses excuse contractual performance when extraordinary events prevent a party from fulfilling obligations. 
Courts have interpreted these clauses narrowly, requiring that: (1) the event be unforeseeable; (2) the event be beyond the party's control; 
(3) the event directly caused the failure to perform.

COVID-19 AND FORCE MAJEURE
Source: Commercial Law Review, Vol. 12 | Page: 2

The COVID-19 pandemic has been extensively litigated as a force majeure event. In JN Contemporary Art LLC v. Phillips Auctioneers LLC 
(S.D.N.Y. 2020), the court held that COVID-19 governmental shutdown orders constituted a 'governmental action' and thus triggered the 
force majeure clause. The court reasoned that the clause listed 'acts of government' and pandemic-related executive orders qualified.

However, in Aukema v. Chesapeake Appalachia LLC, courts noted that general economic hardship or market downturns are typically 
insufficient to trigger force majeure. The party invoking force majeure must demonstrate that performance was objectively impossible, 
not merely more expensive or inconvenient.

FRUSTRATION OF PURPOSE DOCTRINE
Source: Commercial Law Review, Vol. 12 | Page: 3

When force majeure clauses are narrow, parties may invoke frustration of purpose. This doctrine applies when: 
(1) a supervening event occurs after contract formation; (2) the event was unforeseeable; (3) the event destroys the 
principal purpose of the contract; (4) the frustration is not the fault of either party.

In Lloyd v. Murphy (1944), the California Supreme Court held that frustration must be 'substantial' — not merely making 
the contract less profitable. The doctrine is narrowly applied and rarely succeeds when the contract includes a force majeure clause.

GOVERNMENTAL ACTION CLAUSE INTERPRETATION
Source: Commercial Law Review, Vol. 12 | Page: 4

Clauses listing 'governmental action' or 'acts of government' are interpreted to include: executive orders, administrative regulations, 
legislative acts, and emergency declarations. Courts in Gap Inc. v. Ponte Gadea New York LLC held that COVID-19 closure orders issued 
by state governors constituted 'governmental action' sufficient to trigger force majeure.

DAMAGES WHEN FORCE MAJEURE APPLIES
Source: Commercial Law Review, Vol. 12 | Page: 5

When force majeure is successfully invoked, the non-performing party is typically excused from liability for damages. 
The buyer (Alpha Corp) would not be entitled to cover damages, lost profits, or consequential damages during the force majeure period. 
However, if performance becomes permanently impossible, either party may terminate the contract without penalty.
"""
    },
    {
        "filename": "california_employment_law.txt",
        "content": """CALIFORNIA EMPLOYMENT LAW — NON-COMPETE AND IP ASSIGNMENTS
Source: California Employment Law Handbook | Page: 1

CALIFORNIA BUSINESS AND PROFESSIONS CODE SECTION 16600
California Business and Professions Code § 16600 provides: 'Except as provided in this chapter, every contract by which anyone 
is restrained from engaging in a lawful profession, trade, or business of any kind is to that extent void.'

This statute is among the strongest anti-non-compete laws in the United States. In Edwards v. Arthur Andersen LLP (2008), 
the California Supreme Court held that § 16600 bans non-compete agreements broadly, rejecting the 'narrow restraint' exception 
previously recognized by lower courts. Non-compete clauses are void and unenforceable in California regardless of consideration.

CHOICE OF LAW AND CALIFORNIA PUBLIC POLICY
Source: California Employment Law Handbook | Page: 2

Even when an employment agreement contains a choice-of-law clause selecting another state's law, California courts will 
apply California law to non-compete disputes involving California employees. In Application Group Inc. v. Hunter Group Inc., 
the court held that California has a 'strong public policy' against non-competes that overrides foreign choice-of-law provisions.

A Texas choice-of-law provision would not override California § 16600 protections for a California-based employee. 
The employee's right to work freely is a fundamental California public policy interest.

CALIFORNIA LABOR CODE §§ 2870-2872 — EMPLOYEE INVENTIONS
Source: California Employment Law Handbook | Page: 3

California Labor Code § 2870 limits employer IP assignment clauses. Employers CANNOT claim ownership of inventions that:
(1) Do not relate to the employer's business or anticipated R&D;
(2) Do not result from work performed for the employer;
(3) Are developed entirely on the employee's own time with own equipment and resources.

All four conditions must be met for the § 2870 carve-out to apply. If an employee develops an ML algorithm on a personal laptop 
during evenings, unrelated to employer's business, using no employer resources, the invention belongs to the employee.

IP ASSIGNMENT CLAUSE OVERBREADTH
Source: California Employment Law Handbook | Page: 4

An IP assignment clause that purports to assign 'all inventions conceived during employment' without the § 2870 carve-out 
is overbroad and void to that extent under California law. Employers must include statutory carve-out language. 
Courts will sever overbroad provisions rather than enforce them entirely.

INJUNCTIVE RELIEF FOR NON-COMPETE VIOLATIONS
Source: California Employment Law Handbook | Page: 5

Since non-competes are void in California, no injunctive relief is available to enforce them. 
AB 2276 (2023) further strengthened protections by imposing penalties on employers who attempt to enforce void non-competes.
"""
    },
    {
        "filename": "gdpr_data_protection.txt",
        "content": """GDPR COMPLIANCE GUIDE — DATA PROCESSORS AND SECURITY OBLIGATIONS
Source: EU Data Protection Handbook | Page: 1

GDPR ARTICLE 4 — DATA PROCESSOR DEFINITION
A 'data processor' means a natural or legal person, public authority, agency, or other body which processes personal data 
on behalf of the controller. A 'joint controller' exists when two or more controllers jointly determine the purposes and 
means of processing (Article 26).

A cloud vendor operating under a Data Processing Agreement (DPA) is typically a data processor, not a joint controller, 
unless it independently determines the purposes of processing. Under the standard DPA arrangement described, 
the cloud vendor would be classified as a data processor.

GDPR ARTICLE 32 — SECURITY OF PROCESSING
Source: EU Data Protection Handbook | Page: 2

Article 32 requires controllers and processors to implement 'appropriate technical and organisational measures' to ensure 
security appropriate to the risk, including:
(a) Pseudonymisation and encryption of personal data;
(b) Ongoing confidentiality, integrity, availability, and resilience of systems;
(c) Ability to restore data after incidents;
(d) Process for regularly testing security measures.

The term 'industry-standard security measures' in a DPA is ambiguous and may not satisfy Article 32's specificity requirements. 
Supervisory authorities (e.g., Ireland's DPC, UK's ICO) have ruled that vague security language does not meet Article 32 requirements. 
Encryption should be explicitly specified.

GDPR ARTICLE 83 — ADMINISTRATIVE FINES
Source: EU Data Protection Handbook | Page: 3

Article 83(4) imposes fines up to €10 million or 2% of global annual turnover (whichever is higher) for processor violations.
Article 83(5) imposes fines up to €20 million or 4% of global annual turnover for more serious infringements including 
violations of basic processing principles and data subject rights.

A breach exposing 500,000 records would likely be assessed under Article 83(5), with fines up to €20M or 4% of global turnover.

LIABILITY CAPS IN DPAs
Source: EU Data Protection Handbook | Page: 4

Contractual liability caps between controller and processor do not affect supervisory authority fines. 
However, they may limit the controller's ability to seek indemnification from the processor for damages paid to data subjects. 
Whether a liability cap applies depends on whether the processor's negligence contributed to the breach.

DATA BREACH NOTIFICATION
Source: EU Data Protection Handbook | Page: 5

Article 33 requires notification to supervisory authority within 72 hours of becoming aware of a personal data breach. 
Article 34 requires notification to data subjects when the breach is likely to result in high risk to their rights and freedoms. 
Failure to notify is a separate GDPR violation subject to additional fines.
"""
    },
    {
        "filename": "liquidated_damages_cases.txt",
        "content": """LIQUIDATED DAMAGES VS PENALTY CLAUSES — COMPARATIVE ANALYSIS
Source: Contract Law Quarterly | Page: 1

ENGLISH LAW — CAVENDISH SQUARE TEST
In Cavendish Square Holding BV v. Makdessi [2015] UKSC 67, the UK Supreme Court reformulated the penalty clause rule. 
The test is whether the clause imposes a detriment on the contract breaker 'out of all proportion' to any legitimate interest 
of the innocent party in the enforcement of the primary obligation. This replaced the older 'genuine pre-estimate of loss' test 
from Dunlop Pneumatic Tyre Co v. New Garage & Motor Co [1915].

AMERICAN LAW — UNIFORM COMMERCIAL CODE
Source: Contract Law Quarterly | Page: 2

Under UCC § 2-718 and common law, a liquidated damages clause is enforceable if:
(1) Actual damages were difficult to estimate at the time of contracting; AND
(2) The stipulated amount is a reasonable forecast of compensatory damages.

If either condition fails, the clause is an unenforceable penalty. Courts look to whether the clause represents a 
good-faith attempt to anticipate loss, not a punishment for breach.

$50/PACKAGE/DAY ANALYSIS
Source: Contract Law Quarterly | Page: 3

For the e-commerce hypothetical ($50/package/day during peak holiday season), enforceability depends on:
- Whether lost holiday sales were genuinely difficult to calculate at contracting (likely yes — dependent on consumer behavior)
- Whether $50/day represents a reasonable estimate of actual loss (lost sale, customer goodwill, reputational damage)

Courts have enforced per-unit, per-day delay damages in logistics contracts when the commercial context justifies them. 
See UPS Supply Chain Solutions v. Megatrux Transportation.

ACTUAL DAMAGES MITIGATION
Source: Contract Law Quarterly | Page: 4

Even if liquidated damages are upheld, the non-breaching party has a duty to mitigate. If the platform could have sourced 
alternative logistics for 8,000 packages and failed to do so, courts may reduce the award. 
The $2,000,000 total must be weighed against whether mitigation was attempted.
"""
    },
    {
        "filename": "trade_secrets_whistleblower.txt",
        "content": """TRADE SECRETS AND WHISTLEBLOWER PROTECTIONS
Source: Federal Employment & IP Law Review | Page: 1

DEFEND TRADE SECRETS ACT (DTSA) — 18 U.S.C. § 1836
The DTSA provides a federal civil cause of action for trade secret misappropriation. A trade secret includes:
(1) Information that derives economic value from not being generally known; AND
(2) Subject to reasonable measures to maintain secrecy.

Customer lists, pricing structures, and deal pipeline information have been held to qualify as trade secrets when:
- The company took reasonable steps to protect the information (access controls, NDAs, confidentiality training)
- The information was not readily ascertainable by competitors
See Fail-Safe LLC v. A.O. Smith Corp. (E.D. Wis. 2010).

MISAPPROPRIATION BY DEPARTING EMPLOYEES
Source: Federal Employment & IP Law Review | Page: 2

Emailing confidential data to a personal account before departing constitutes misappropriation under the DTSA. 
In PQ Labs Inc. v. Yang Qi, the court held that downloading trade secret data to personal devices days before resignation, 
then using that data at a competitor, constituted willful and malicious misappropriation justifying enhanced damages.

REMEDIES UNDER DTSA
Source: Federal Employment & IP Law Review | Page: 3

Available remedies include:
(1) Injunctive relief — prevent use or disclosure of the trade secret;
(2) Compensatory damages — actual loss or unjust enrichment;
(3) Exemplary damages — up to 2x compensatory damages for willful misappropriation;
(4) Attorney's fees — if misappropriation was willful and malicious.

WHISTLEBLOWER RETALIATION — SARBANES-OXLEY
Source: Federal Employment & IP Law Review | Page: 4

Section 806 of the Sarbanes-Oxley Act (SOX) protects employees of publicly traded companies who report violations of 
securities laws, SEC rules, or fraud. Key requirements:
(1) The employer must be a publicly traded company or a subsidiary/contractor thereof;
(2) The employee engaged in protected activity (reporting to supervisor, law enforcement, or Congress);
(3) The employer knew of the protected activity;
(4) The discharge or adverse action was caused by the protected activity.

DODD-FRANK WHISTLEBLOWER PROTECTIONS
Source: Federal Employment & IP Law Review | Page: 5

Dodd-Frank Section 922 extends protections to employees who report securities violations to the SEC. Importantly, 
the Supreme Court in Digital Realty Trust v. Somers (2018) held that Dodd-Frank protections apply ONLY to employees 
who report externally to the SEC, not to internal reports. For a private company, SOX protections are limited; 
Dodd-Frank does not apply unless the company is publicly traded or the employee reports to the SEC.
"""
    }
]

def generate_sample_data(output_dir: str = "data/"):
    Path(output_dir).mkdir(exist_ok=True)
    for case in SAMPLE_CASES:
        path = Path(output_dir) / case["filename"]
        path.write_text(case["content"], encoding="utf-8")
        print(f"[Data] Created {path}")
    print(f"[Data] Generated {len(SAMPLE_CASES)} sample legal documents in {output_dir}/")

if __name__ == "__main__":
    generate_sample_data()
