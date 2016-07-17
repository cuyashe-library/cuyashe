#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE cuYASHE
#include <boost/test/unit_test.hpp>
#include <NTL/ZZ_pEX.h>
#include "../settings.h"
#include "../aritmetic/polynomial.h"
#include "../distribution/distribution.h"
#include "../yashe/yashe.h"
#include "../yashe/ciphertext.h"

#define NTESTS 100

struct AritmeticSuite
{
	Distribution dist;
    ZZ q;
    bn_t Q;
    poly_t phi;
    ZZ_pX NTL_Phi;
    bn_t UQ;

	// Test aritmetic functions
    AritmeticSuite(){
        // Log
        log_init("aritmetic_test.log");

        // Init
        OP_DEGREE = 32;
        int mersenne_n = 127;
        q = NTL::power2_ZZ(mersenne_n) - 1;
        get_words(&Q,q);
        UQ = get_reciprocal(q);
        gen_crt_primes(q,OP_DEGREE);
        CUDAFunctions::init(OP_DEGREE);
        poly_init(&phi);
        poly_set_nth_cyclotomic(&phi,2*OP_DEGREE);
        poly_print(&phi);
        std::cout << "Degree: " << poly_get_deg(&phi) << std::endl;

        // Init NTL
        ZZ_p::init(q);
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

struct YasheSuite
{
    Distribution dist;
    ZZ q;
    cuyasheint_t t;
    poly_t phi;
    Yashe *cipher;

    // Test Yashe functions
    YasheSuite(){
        // Log
        log_init("yashe_test.log");

        // Init
        //OP_DEGREE = 4096;
        OP_DEGREE = 32;
        // OP_DEGREE = 2048;
        int mersenne_n = 127;
        q = NTL::power2_ZZ(mersenne_n) - 1;
        // t = 17;
        t = 1024;
        ZZ w = NTL::power2_ZZ(32);

        gen_crt_primes(q,OP_DEGREE);
        CUDAFunctions::init(OP_DEGREE);
        poly_init(&phi);
        poly_set_nth_cyclotomic(&phi,2*OP_DEGREE);
        poly_print(&phi);
        BOOST_TEST_MESSAGE("Degree: " << poly_get_deg(&phi));

        // Init NTL
        ZZ_p::init(q);
        ZZ_pX NTL_Phi;
        for(int i = 0; i <= poly_get_deg(&phi);i++)
          NTL::SetCoeff(NTL_Phi,i,conv<ZZ_p>(poly_get_coeff(&phi,i)));

        ZZ_pE::init(NTL_Phi);

        // Object used to generate random elements
        dist = Distribution(UNIFORMLY);

        // YASHE parameters
        cipher = new Yashe();

        Yashe::nphi = poly_get_deg(&phi);
        Yashe::nq = mersenne_n;

        poly_set_coeff(&Yashe::t,0,to_ZZ(t));
        Yashe::w = w;
        Yashe::lwq = floor(NTL::log(q)/NTL::log(to_ZZ(w)))+1;

        cipher->generate_keys();
    }

    ~YasheSuite()
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
    for(int ntest = 0; ntest < NTESTS; ntest++){
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

        poly_free(&a);
    }
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

    poly_free(&a);
    poly_free(&b);
    poly_free(&c);

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
    poly_reduce(&c,OP_DEGREE,Q,NTL::NumBits(q));
    ZZ_pEX ntl_c = ntl_a*ntl_b % conv<ZZ_pEX>(NTL_Phi);

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

    poly_free(&a);
    poly_free(&b);
    poly_free(&c);

}

BOOST_AUTO_TEST_CASE(add_mul)
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
    
    poly_t c;
    poly_init(&c);
    ZZ_pEX ntl_c;
    // Add
    poly_add(&c,&a,&b);
    ntl_c = ntl_a + ntl_b;

    // Mul
    poly_mul(&c,&c,&b);
    ntl_c = ntl_c * ntl_b;

    BOOST_CHECK_EQUAL(NTL::deg(ntl_c),poly_get_deg(&c));
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

    poly_free(&a);
    poly_free(&b);
    poly_free(&c);
}

BOOST_AUTO_TEST_CASE(simpleReduce)
{
    for(int count = 0; count < NTESTS; count++){
        poly_t a;
        poly_init(&a);

        poly_set_coeff(&a,0,to_ZZ("0"));
        poly_set_coeff(&a,1,to_ZZ("0"));
        poly_set_coeff(&a,2,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,3,to_ZZ("0"));
        poly_set_coeff(&a,4,to_ZZ("0"));
        poly_set_coeff(&a,5,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,6,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,7,to_ZZ("0"));
        poly_set_coeff(&a,8,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,9,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,10,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,11,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,12,to_ZZ("0"));
        poly_set_coeff(&a,13,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,14,to_ZZ("0"));
        poly_set_coeff(&a,15,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,16,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,17,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,18,to_ZZ("0"));
        poly_set_coeff(&a,19,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,20,to_ZZ("0"));
        poly_set_coeff(&a,21,to_ZZ("0"));
        poly_set_coeff(&a,22,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,23,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,24,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,25,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,26,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,27,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,28,to_ZZ("0"));
        poly_set_coeff(&a,29,to_ZZ("0"));
        poly_set_coeff(&a,30,to_ZZ("174054430680060024061516111701349440158720"));
        poly_set_coeff(&a,31,to_ZZ("0"));
        poly_set_coeff(&a,32,to_ZZ("174054430680060024061516111701349440158720"));
   
        poly_reduce(&a,OP_DEGREE,Q,NTL::NumBits(q));

        if(OP_DEGREE == 32){
            ZZ expected_result[] = {to_ZZ("1"), to_ZZ("0"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("0"), to_ZZ("0"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("0"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("0"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("0"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("0"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("0"), to_ZZ("0"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("0"), to_ZZ("0"), to_ZZ("170141183460469231731687303715884105726"), to_ZZ("0")};
            
            for(int i = 0; i < OP_DEGREE;i++)
                BOOST_CHECK_EQUAL(poly_get_coeff(&a,i),expected_result[i]);
            
        }else{
            throw "";
        }

        poly_free(&a);
    }

}
BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(YasheFixture, YasheSuite)

BOOST_AUTO_TEST_CASE(simple_encryptdecrypt)
{
    
    const int i = 42;
    std::cout << "Testing " << i << std::endl;

    poly_t m;
    poly_init(&m);
    poly_set_coeff(&m,0,to_ZZ(i));

    cipher_t c;
    cipher_init(&c);
    cipher->encrypt(&c,m); //

    poly_t m_decrypted;
    poly_init(&m_decrypted);
    cipher->decrypt(&m_decrypted,c); //

    BOOST_CHECK_EQUAL(poly_get_coeff(&m,0) , poly_get_coeff(&m_decrypted, 0));
    
    poly_free(&m);
    poly_free(&m_decrypted);
}


BOOST_AUTO_TEST_CASE(simple_add)
{
    
    const int i = 42;
    const int j = 13;

    // Messages
    // 
    poly_t mi;
    poly_init(&mi);
    poly_set_coeff(&mi,0,to_ZZ(i));

    poly_t mj;
    poly_init(&mj);
    poly_set_coeff(&mj,0,to_ZZ(j));

    // Encrypt
    // 
    cipher_t ci;
    cipher_init(&ci);
    cipher->encrypt(&ci,mi); //

    cipher_t cj;
    cipher_init(&cj);
    cipher->encrypt(&cj,mj); //

    // Addition
    // 
    cipher_t cz;
    cipher_init(&cz);
    cipher_add(&cz,&ci,&cj);

    poly_t m_decrypted;
    poly_init(&m_decrypted);
    cipher->decrypt(&m_decrypted,cz); //

    BOOST_CHECK_EQUAL( i+j , poly_get_coeff(&m_decrypted, 0));
    
    poly_free(&mi);
    poly_free(&mj);
    poly_free(&m_decrypted);
    cipher_free(&ci);
    cipher_free(&cj);
    cipher_free(&cz);
}

BOOST_AUTO_TEST_CASE(add)
{
    for(int n = 0; n < NTESTS; n++ ){

        const ZZ i = NTL::RandomBnd(to_ZZ(t));
        const ZZ j = NTL::RandomBnd(to_ZZ(t));

        // Messages
        // 
        poly_t mi;
        poly_init(&mi);
        poly_set_coeff(&mi,0,i);

        poly_t mj;
        poly_init(&mj);
        poly_set_coeff(&mj,0,j);

        // Encrypt
        // 
        cipher_t ci;
        cipher_init(&ci);
        cipher->encrypt(&ci,mi); //

        cipher_t cj;
        cipher_init(&cj);
        cipher->encrypt(&cj,mj); //

        // Addition
        // 
        cipher_t cz;
        cipher_init(&cz);
        cipher_add(&cz,&ci,&cj);

        poly_t m_decrypted;
        poly_init(&m_decrypted);
        cipher->decrypt(&m_decrypted,cz); //

        BOOST_CHECK_EQUAL( (i+j) % to_ZZ(t) , poly_get_coeff(&m_decrypted, 0));
        
        poly_free(&mi);
        poly_free(&mj);
        poly_free(&m_decrypted);
        cipher_free(&ci);
        cipher_free(&cj);
        cipher_free(&cz);
    }
}

BOOST_AUTO_TEST_CASE(simple_mul)
{
    
    const int i = 42;
    const int j = 13;

    // Messages
    // 
    poly_t mi;
    poly_init(&mi);
    poly_set_coeff(&mi,0,to_ZZ(i));

    poly_t mj;
    poly_init(&mj);
    poly_set_coeff(&mj,0,to_ZZ(j));

    // Encrypt
    // 
    cipher_t ci;
    cipher_init(&ci);
    cipher->encrypt(&ci,mi); //
    log_debug("c1: " + poly_print(&ci.p));


    cipher_t cj;
    cipher_init(&cj);
    cipher->encrypt(&cj,mj); //
    log_debug("c2: " + poly_print(&cj.p));

    // Multiplication
    // 
    cipher_t cz;
    cipher_init(&cz);
    cipher_mul(&cz,&ci,&cj);

    poly_t m_decrypted;
    poly_init(&m_decrypted);
    cipher->decrypt(&m_decrypted,cz); //

    BOOST_CHECK_EQUAL( i*j , poly_get_coeff(&m_decrypted, 0));
    
    poly_free(&mi);
    poly_free(&mj);
    poly_free(&m_decrypted);
    cipher_free(&ci);
    cipher_free(&cj);
    cipher_free(&cz);
}
BOOST_AUTO_TEST_CASE(mul)
{
    for(int n = 0; n < NTESTS; n++){

        const ZZ i = NTL::RandomBnd(to_ZZ(t));
        const ZZ j = NTL::RandomBnd(to_ZZ(t));

        // Messages
        // 
        poly_t mi;
        poly_init(&mi);
        poly_set_coeff(&mi,0,to_ZZ(i));

        poly_t mj;
        poly_init(&mj);
        poly_set_coeff(&mj,0,to_ZZ(j));

        // Encrypt
        // 
        cipher_t ci;
        cipher_init(&ci);
        cipher->encrypt(&ci,mi); //

        cipher_t cj;
        cipher_init(&cj);
        cipher->encrypt(&cj,mj); //

        // Multiplication
        // 
        cipher_t cz;
        cipher_init(&cz);
        cipher_mul(&cz,&ci,&cj);

        poly_t m_decrypted;
        poly_init(&m_decrypted);
        cipher->decrypt(&m_decrypted,cz); //

        BOOST_CHECK_EQUAL( i*j % to_ZZ(t) , poly_get_coeff(&m_decrypted, 0));
        
        poly_free(&mi);
        poly_free(&mj);
        poly_free(&m_decrypted);
        cipher_free(&ci);
        cipher_free(&cj);
        cipher_free(&cz);
    }
}

BOOST_AUTO_TEST_CASE(encryptdecrypt)
{
 
    for(int i = 2; i < NTESTS; i++){
        long value = NTL::RandomWord() % 1024;

        poly_t m;
        poly_init(&m);
    	// std::cout << "Testing " << value << std::endl;
    	
        poly_set_coeff(&m,0,to_ZZ(value));

        cipher_t c;
        cipher_init(&c);
        cipher->encrypt(&c,m); //

        poly_t m_decrypted;
        poly_init(&m_decrypted);
        cipher->decrypt(&m_decrypted,c); //

        BOOST_CHECK_EQUAL(poly_get_coeff(&m,0) , poly_get_coeff(&m_decrypted, 0));
        
        poly_free(&m);
        poly_free(&m_decrypted);
        cipher_free(&c);
    }
}
BOOST_AUTO_TEST_SUITE_END()
