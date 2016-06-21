#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE cuYASHE
#include <boost/test/unit_test.hpp>
#include <NTL/ZZ_pEX.h>
#include "../settings.h"
#include "../aritmetic/polynomial.h"
#include "../distribution/distribution.h"


#define NTESTS 10

struct AritmeticSuite
{
	Distribution dist;
    ZZ q;
    poly_t phi;

	// Test aritmetic functions
    AritmeticSuite(){
        // Log
        log_init("test.log");

        // Init
        OP_DEGREE = 32;
        q = to_ZZ("17");
        gen_crt_primes(q,OP_DEGREE);
        CUDAFunctions::init(OP_DEGREE);
        poly_init(&phi);
        poly_set_nth_cyclotomic(&phi,2*OP_DEGREE);
        poly_print(&phi);
        std::cout << "Degree: " << poly_get_deg(&phi) << std::endl;

        // Init NTL
        ZZ_p::init(q);
        ZZ_pX NTL_Phi;
        for(int i = 0; i <= poly_get_deg(&phi);i++)
          NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(poly_get_coeff(&phi,i)));

        ZZ_pE::init(NTL_Phi);

        // Object used to generate random elements
    	dist = Distribution(UNIFORMLY);
    }

    ~AritmeticSuite()
    {
        BOOST_TEST_MESSAGE("teardown mass");
        cudaDeviceReset();
    }
};


BOOST_FIXTURE_TEST_SUITE(AritmeticFixture, AritmeticSuite)

BOOST_AUTO_TEST_CASE(set_coeff)
{
    poly_t a;
    poly_init(&a);

    poly_set_coeff(&a,0,to_ZZ("10"));

    BOOST_CHECK_EQUAL(poly_get_coeff(&a,0) , to_ZZ("10"));
}

BOOST_AUTO_TEST_CASE(crt)
{
    poly_t a;
    poly_init(&a);
    BOOST_TEST_CHECKPOINT("Setting up polynomial");
    for(int i = 0; i < OP_DEGREE;i++)
        poly_set_coeff(&a,i,to_ZZ(i*i));

    // Elevate to TRANSTATE
    BOOST_TEST_CHECKPOINT("Elevating");
    while(a.status != TRANSSTATE)
        poly_elevate(&a);
    
    // Demote back to HOST
    BOOST_TEST_CHECKPOINT("Demoting");
    while(a.status != HOSTSTATE)
        poly_demote(&a);

    for(int i = 0; i < OP_DEGREE;i++)
        BOOST_CHECK_EQUAL(poly_get_coeff(&a,i) , to_ZZ(i*i));
}

BOOST_AUTO_TEST_CASE(add)
{   
    // Init
    BOOST_TEST_CHECKPOINT("Initiating");
    poly_t a,b;
	poly_init(&a);
	poly_init(&b);

    // Sample
    BOOST_TEST_CHECKPOINT("Will sample");
  	dist.generate_sample(&a, 50, OP_DEGREE);
  	dist.generate_sample(&b, 50, OP_DEGREE);

    // Add
    BOOST_TEST_CHECKPOINT("adding...");
    poly_t c;
    poly_init(&c);
    
    poly_add(&c, &a, &b);

    // Compare
    for(int i = 0; i < poly_get_deg(&c); i++){
        ZZ x = poly_get_coeff(&a,i);
        ZZ y = poly_get_coeff(&b,i);
        ZZ z = poly_get_coeff(&c,i);

        BOOST_CHECK_EQUAL(x+y,z);
    }

}

BOOST_AUTO_TEST_CASE(mul)
{  
    ZZ_pEX ntl_a;
    ZZ_pEX ntl_b;
    poly_t a;
    poly_t b;
    
    // Init
    poly_init(&a);
    poly_init(&b);
    dist.generate_sample(&a,5,OP_DEGREE);
    dist.generate_sample(&b,5,OP_DEGREE);
    for(int i = 0; i < OP_DEGREE; i++){
        NTL::SetCoeff(ntl_a,i,conv<ZZ_p>(poly_get_coeff(&a,i)));
        NTL::SetCoeff(ntl_b,i,conv<ZZ_p>(poly_get_coeff(&b,i)));
    }

    // Mul
    poly_t c;
    poly_init(&c);
    poly_mul(&c,&a,&b);
    ZZ_pEX ntl_c = ntl_a*ntl_b;

    // Verify
    for(int i = 0; i < 2*OP_DEGREE; i++){
        ZZ ntl_value;
        if( NTL::IsZero(NTL::coeff(ntl_c,i)) )
        // Without this, NTL raises an exception when we call rep()
          ntl_value = 0L;
        else
          ntl_value = conv<ZZ>(NTL::rep(NTL::coeff(ntl_c,i))[0]);

        BOOST_CHECK_EQUAL(poly_get_coeff(&c,i) , ntl_value);
    }
}
BOOST_AUTO_TEST_SUITE_END()
