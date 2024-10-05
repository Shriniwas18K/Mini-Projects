import './Events.css';
export default function Events() {
    const events = [
        ["Presentation on AI Tools used in Industry - Winners | Engineer's Day", "./events/winners_presentation_engg_day.jpeg"],
        ["Pixel Perfect Workshop | Photography Cell", "./events/Pixel Perfect Workshop.jpeg"],
        ["Teacher's Day Celebration 2023", "./events/teachers_day.jpeg"],
        ["ARTIMAS 2023 - Winners", "./events/Winners.jpeg"],
        ["Houdini Heist", "./events/Houdini.jpeg"],
        ["Hack Matrix", "./events/hack.jpeg"],
        ["Neuro Drain", "./events/Neuro.jpeg"],
        ["Adrenaline Rush", "./events/Adrenaline.jpeg"],
        ["Cybernetic Visions", "./events/poster.jpeg"],
        ["Pixel Perfect", "./events/photography.jpeg"],
        ["DIY websites", "./events/Build Your Own Website.jpeg"],
        ["AiMSA Induction", "./events/inductions.jpeg"]
    ]
    const events_list = events.map((card, index) => (
        <article key={index} className="event-card">
            <a href={card[1]} alt={card[0]} >
                <img src={card[1]} alt={card[0]} loading='lazy' ></img>
                <p>{card[0]}</p>
            </a>
        </article>
    ))
    return (
        <>
            <section id="Events">
                <img id="event-aimsa-img" src='AIMSAnobg.png' alt='aimsa'/>
                <h1>Events-Gallery</h1>
                {events_list}
            </section>
        </>
    )
}