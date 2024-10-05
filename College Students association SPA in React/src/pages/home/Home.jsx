import React, { useEffect, useRef } from 'react';
import './homestyles.css';

export default function Home(){
  const facultiesRef = useRef(null);
  const atharvaRef = useRef(null);
  const parthRef=useRef(null);
  useEffect(() => {
    const options = {
      root: null,
      threshold: 0
    };

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('fade-in');
        }
      });
    }, options);
    if (facultiesRef.current) {
      observer.observe(facultiesRef.current);
    }
    if (atharvaRef.current) {
      observer.observe(atharvaRef.current);
    }
    if(parthRef.current){
        observer.observe(parthRef.current);
    }

    return () => {
      observer.disconnect(); // Disconnect the observer when component unmounts
    };
  }, []);
    return (
        <>  <section id="banner" className="homepagesections">
            <img src="./homepage/aimsa.jpg" alt="banner" ></img>
            <article id="description">
                AiMSA is a student association that has been established for the benefit of
                students in the CSE AIML branch at PCCOE. Our mission is to provide a
                platform for students to engage in meaningful academic, social, and professional
                activities that will enhance their educational experience. We aim to create an
                environment that fosters learning and growth by organizing workshops, seminars,
                and guest lectures on various topics related to computer science, artificial
                intelligence, and machine learning. Additionally, we facilitate opportunities
                for students to connect with industry professionals and build networks that can
                lead to future career prospects. Our association is committed to promoting
                teamwork, leadership, and innovation among our members, and we strive to make a
                positive impact on the larger community.
            </article>
        </section>
            <section id="home-main" className="homepagesections">
                <h2>Our Mentors</h2>
                <article id="faculties" ref={facultiesRef}>
                    <figure>
                        <img src="./homepage/HOD.jpg" alt="HODimg" loading="lazy"/>
                        <figcaption>
                            <h3>Dr. Anuradha Thakare</h3>
                            <h4>Head of Department <br />CSE(AIML)</h4>
                        </figcaption>
                    </figure>
                    <figure>
                        <img src="./homepage/SDW.jpg" alt="SDWimg" loading="lazy"/>
                        <h3>Prof. Pallavi Dhade</h3>
                        <h4>SDW Co-ordinator<br />CSE(AIML)</h4>
                    </figure>
                </article>
                <h2>Our Heads</h2>
                <article id="commitee" >
                    <figure ref={atharvaRef}>
                        <img src="./homepage/atharva_joshi_president.png" alt="president" loading='lazy' />
                        <figcaption>
                            <h2>Atharva Joshi</h2>
                            <h4>President</h4>
                            <blockquote>I am Atharva Joshi, and it is an immense honor to serve as the President of AiMSA.
                                "We are new, but that doesn't mean we are weak." These words encapsulate the spirit of
                                AiMSA. While we may be a young organization, our determination, passion, and commitment to
                                excellence are unwavering. Our team comprises talented individuals from diverse backgrounds,
                                each bringing unique skills and perspectives to the table.
                                My Team, Team AiMSA is One Unit; A force to be reckoned with and we believe to achieve
                                greater heights with each new milestone we unlock.
                                At AiMSA, we are driven by the vision of a smarter age, where the power of artificial
                                intelligence and cutting-edge technologies revolutionize our way of living.
                                As we navigate this exciting path forward, we promise to uphold the values of transparency,
                                integrity, and inclusivity. Our efforts will be focused on fostering innovation, promoting
                                research, and ensuring that the benefits of AI reach every student.
                                We Plan to co-ordinate and conduct various activities which display the power of AI Today!
                                I invite you to join us on this remarkable journey. Together, we can shape a future where AI
                                serves as a force for good and enhances the lives of people everywhere.
                            </blockquote>
                        </figcaption>
                    </figure>
                    <figure ref={parthRef}>
                        <img src="./homepage/parth_singhal_vice_president.png" alt="vice-president" loading='lazy' />
                        <h2>Parth Singhal</h2>
                        <h4>Vice President</h4>
                        <blockquote>I am honored to serve as the Vice President of PCCOE AiMSA, and I
                            address you with great enthusiasm. Together, we have the power to shape the future of
                            technology. Let us embrace learning, innovation, and exceeding expectations. Our department
                            thrives on diversity, inclusivity, and mutual support. We encourage each other to explore beyond
                            the boundaries of our campus and make a positive impact on society. In the face of challenges,
                            resilience will be our driving force. Together, we will leave an indelible mark on the world of
                            technology and inspire generations to come.</blockquote>
                    </figure>
                </article>
            </section>
        </>)
}