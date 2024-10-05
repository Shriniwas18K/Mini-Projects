import './Team.css';
export default function Team() {
    const records = [
        ['Atharv Joshi', 'President', './profile_images/atharva_joshi_president.png'],
        ['Parth Singhal', 'Vice President', './profile_images/parth_singhal_vice_president.png'],
        ['Parth Chitale', 'Managing Body Chairperson', './profile_images/parth_chitale_managing_body_chairperson.png'],
        ['Lokesh Kad', 'Secretary', './profile_images/lokesh_kad_secretary.png'],
        ['Shubham Narsale', 'Secretary', './profile_images/soham_narsale_secretary.png'],
        ['Vaibhav Gangurde', 'Treasurer', './profile_images/vaibhav_gangrude_treasurer.png'],
        ['Shubham Singh', 'Treasurer', './profile_images/shubham_singh_treasurer.png'],
        ['Prathemesh Kolekar', 'Webmaster', './profile_images/prathmesh_kolekar_webmaster.png'],
        ['Parth Halawane', 'Webmaster', './profile_images/parth_halwane_webmaster.png'],
        ['Om Bhavsar', 'Designer', './profile_images/om_bhavsar_design.png'],
        ['Tushar Garad', 'Designer', './profile_images/tushar_garad_design.png'],
        ['Tushar Mahajan', 'Photography', './profile_images/tushar_mahajan_photography.png'],
        ['Harshad Gholve', 'Photography', './profile_images/harshad_gholve_photography.png'],
        ['Anurag Karpe', 'Photography', './profile_images/anurag_karpe_photography.png'],
        ['Anurag Mishra', 'Social Media', './profile_images/anurag_mishra_social_media.png'],
        ['Aarya Hire', 'Marketing', './profile_images/aarya_hire_marketing.png'],
        ['Jivan Karande', 'Marketing', './profile_images/jivan_karande_marketing.png'],
    ]
 
    const record_list = records.map((record, index) => (
        <button className="teamcard" key={index}>
            <span></span>
            <img className="teamcardprofileimg" src={record[2]} alt={record[0]} loading='lazy' />
            <h3>{record[0]}</h3>
            <h4>{record[1]}</h4>
            <article className='teamcard-links'>
                <a href={record[3]}><img src='./profile_images/instagramicon.png' alt={record[0] + ' instagram link'}></img> </a>
                <a href={record[4]}><img src='./profile_images/linkedinicon.png' alt={record[0] + ' linkedin link'}></img></a>
                <a href={record[5]}><img src='./profile_images/gmailicon.png' alt={record[0] + ' gmail'}></img></a>
            </article>
        </button>
    ));
    return (
        <>  <a href='https://team3daimsa.vercel.app/'><button id='veiw3d'>VEIW 3D</button></a>
            <section id="Team">
                {record_list}
            </section>
        </>
    )
}